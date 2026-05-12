# Audit Stage 28.9 cycle 87 — Code review

Scope: HEAD `d8e5807`.

## Verdict

**PASS — 0 findings at confidence >= 75%.**

## Review areas

### 1. Cycle-86 code-quality: docstring updates accurate vs new behavior?

- Comment claim "Default 64 for unrecognized types preserves prior behavior"
  matches the pre-fix hard-coded bound of 64 — verified by diff inspection.
- Error message format `[0, {_bits}) for {_bits}-bit SHL fold` is half-open
  and matches the runtime check `r < 0 or r >= _bits`. Internally consistent.
- Both SHL and SHR carry equivalent fix-comments with cross-reference
  ("see SHL note above"). No drift.

### 2. Regression tests test_c85_1_* discriminative power

- `test_c85_1_shift_bound_uses_result_type_bitwidth`: pre-fix would NOT
  raise (`32 >= 64` is False → silent fold); test hits the
  `AssertionError` tail. Post-fix raises `ShiftFoldError` and short-
  circuits at the `return`. Discriminative — fails under regression.
- `test_c85_1_shift_i64_still_allows_up_to_63`: pre-fix and post-fix
  both pass (i64 bound is 64 in both worlds). Test is explicitly
  labeled "regression boundary preservation" — not discriminative for
  the bug, by design. Acceptable.

### 3. Other hardcoded 32/64 in const_fold.py that should be type-derived?

- `_wrap_int_to_type` (lines 121, 123) defaults bits=32 for unknown
  scalar / non-scalar types. Pre-existing across many cycles and a
  deliberate "unknown ⇒ narrow" choice; not introduced or perturbed
  by cycle-86. Per scope rules, deferred-known not re-flagged.
- No other hardcoded shift-width / wrap-width literals appear in
  const_fold.py after line 130.

### 4. Test runner __main__ pattern survey

- Surveyed all 35 test files in `helixc/tests/test_*.py` carrying a
  `if __name__ == "__main__":` block.
- 12 delegate to `pytest.main([__file__, "-v"])` — auto-discovery.
- 23 call a local `main()` function.
- All 23 `main()` implementations build the test list via
  `[(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]` — the same auto-discovery pattern used by the cycle-84 `test_ffi.py` fix.
- No remaining test file carries a hard-coded test list in `__main__`.

## Conclusion

No findings at confidence >= 75%. Stage 28.9 cycle 87 PASS.
No edits made. Read-only audit.
