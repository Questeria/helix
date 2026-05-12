# Audit Stage 28.9 cycle 85 — Code review

**Scope**: HEAD `fb80a4f` (Stage 28.9 only; Stage 28.10 commits explicitly out of scope).
**Mode**: STRICT READ-ONLY (Read/Grep/Glob/Bash only; one Write to this file; no Edit).
**Result**: **PASS** — 0 findings at confidence >= 75%.

---

## Deferred-known / prior-flagged items NOT re-flagged

- `test_ast_walker.py` and `test_codegen_determinism.py` have no `__main__`
  block — declared out-of-scope from the cycle-84 regression class in
  `audit-stage28-9-cycle85-silent-failures.md` lines 37-39. NOT re-flagged.
- All C1-C84 prior code-review findings + deferred items: NOT re-flagged.

---

## Cycle-84 fix completeness

`helixc/tests/test_ffi.py:181-195` — the runner now uses globals() discovery.
The replacement carries a 9-line in-source comment that:
- attributes the change ("Stage 28.9 cycle 84 audit-CR CR-1 fix (HIGH conf 90)");
- describes the pre-fix bug ("this list was hard-coded with the 3 Stage-16.5
  tests and the cycle-77/79/81 regression test ... was silently omitted");
- names the affected gate path (`scripts/run_all_tests.sh` invoking
  `python helixc/tests/test_ffi.py`);
- references the reference pattern (`test_ir.py / test_totality.py`);
- states the forward-compatibility guarantee ("any future `def test_*` is
  auto-picked-up").

The docstring/comment is clear, attributable, and self-contained. **Verified PASS.**

---

## Survey: other test_*.py `__main__` runners for the same hard-coded pattern

Scanned all 37 `helixc/tests/test_*.py` files. Three classes:

1. **`main()` with globals() discovery** (24 files): test_ast_hash, test_autodiff,
   test_autodiff_parity, test_autodiff_reverse, test_codegen, test_const_fold,
   test_cse, test_dce, test_effect_check, test_fdce, test_ffi (post cycle-84),
   test_hash_cons, test_ir, test_lexer, test_match, test_parser, test_presburger,
   test_ptx, test_reflection, test_select_codegen, test_strings_io, test_tile_ir,
   test_totality, test_transcendentals, test_typecheck. Auto-discover any
   `def test_*` — same forward-compat guarantee as the cycle-84 fix. **Clean.**

2. **`pytest.main([__file__, ...])`** (10 files): test_autotune, test_cli,
   test_deprecated, test_diagnostics, test_panic, test_provenance, test_pytree,
   test_struct_mono, test_trace, test_unsafe. pytest auto-collects every
   `def test_*` function in the file — at least as thorough as globals(). **Clean.**

3. **No `__main__` block** (2 files): test_ast_walker, test_codegen_determinism.
   Already addressed in cycle-85 silent-failures doc (lines 37-39) as
   out-of-scope from the cycle-84 regression class. **Not re-flagged.**

No file in scope is missing tests that should be in the heavy gate.

---

## helixc/check.py CLI behavior consistency between flags

Flag inventory (`_KNOWN_LONG_FLAGS` + `parse_args` dispatch at lines 81-153):
`--help`/`-h`, `--no-color`, `--color`, `--stdlib`, `--hash`, `--hash-cons`,
`--strict`, `--check-only`, `--emit-ast`, `--emit-ir`, `--emit-asm`,
`--emit-ptx`, `--doc`, `-O0..3`, `-o <path>`, `-l <libname>`, `-W<flag>[=<val>]`.

- **Error vs warning policy split**: panic/unwind/trace/unsafe/autotune/struct-mono
  diags return rc=1 unconditionally (malformed source — design-correct ERROR class).
  deprecated/totality/ad emit warnings and only return rc=1 when promoted by
  `--strict` / `-Wdeprecated=error` / `-Wad=error`. Consistent with the
  in-line documentation at lines 596-609 (cycle-26 audit-R C25-1 NOTE). Clean.

- **AD-warning drain universality**: `main()` wraps `_main_inner` in try/finally
  (lines 284-338) so every return path drains. `_drain_ad_init()` runs at the
  start of every main invocation (line 275) so cross-call state stays clean.
  Pre-banner short-circuits (`--help`, `--doc`, bad invocation) skip the drain
  intentionally because typecheck never runs. Consistent with cycle-2 C2-1 doc.

- **--check-only short-circuit position** (line 610): after totality / deprecated
  / trace / panic / unsafe / autotune; before lower/fold/cse/dce/effect_check.
  Intentional per cycle-26 C25-1 NOTE — documented in 14-line in-source comment.

- **Help-text `-W<flag>` example** (line 36): cites `-Wdeprecated`, doesn't cite
  `-Wad`. Both are recognized at runtime (lines 236, 512). Minor doc-completeness
  gap; below conf 75% — not flagged as a CLI-behavior inconsistency.

- **Errors-list short-circuit** (`_main_inner` lines 354-357): `--help` is consumed
  BEFORE the errors-list bail, so `helixc --help bogus-flag` prints help and
  returns 0 instead of failing on the unknown flag. Consistent with standard
  CLI convention (POSIX `--help` is highest-precedence). Clean.

No CLI consistency findings at conf >= 75%.

---

## Verdict

**PASS — 0 findings at conf >= 75%.**

Counter: cycle 85 (0 code-review findings) PASS → counter advances toward 5-clean.

**No edits applied. This document is the sole write artifact.**
