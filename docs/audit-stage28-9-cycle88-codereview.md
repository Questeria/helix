# Audit Stage 28.9 cycle 88 — Code review

**Scope** HEAD e0967670d5c959444ce8d8d09b38e396b7a348ff (`Stage 28.9 cycle-87 audit clean (1/5 after cycle-85 reset)`).

**Mode** Strict read-only. Read / Grep / Glob / Bash only. ONE Write (this file). NO Edit.

## Scope (narrow)

- `helixc/check.py` CLI flag handling — completeness for `-W`, `-O`, `-l`, `--doc` behavior.
- `helixc/tests/test_*.py` — duplicate test names, conflicting docstrings.
- Cycle-86 follow-on cleanup (e.g. comment drift in `helixc/ir/passes/const_fold.py` and `helixc/tests/test_const_fold.py`).

Prior C1–C87 findings + the deferred-known list are NOT re-flagged.

## Result

**FAIL — 1 finding at conf >= 75%** (covers 4 occurrences of the same defect class).

---

## C88-1 — Duplicate `def` shadows earlier test bodies in `test_codegen.py` (HIGH, conf 95)

`helixc/tests/test_codegen.py` defines four `test_stdlib_vec_*` functions twice each in the same module. Python `def` semantics rebind the name on the second definition, so pytest only collects the second body; the first body is dead code that never executes. Coverage is overstated by 4 cases against the 585-test count in the project memo. The duplicated names with line numbers:

| Test name | First `def` | Second `def` (winner) | Bodies differ? |
|-----------|-------------|-----------------------|----------------|
| `test_stdlib_vec_eq` | line 8331 | line 13443 | YES — first calls `vec_eq(a, b, 3)`, second calls `vec_eq(a, 3, b, 3)` (different arity) |
| `test_stdlib_vec_first` | line 11116 | line 12734 | YES — first uses `__arena_push`, second uses `vec_push` and also checks empty case |
| `test_stdlib_vec_last` | line 11129 | line 12751 | YES — same pattern: arena-based vs vec-push-based + empty check |
| `test_stdlib_vec_reverse_inplace` | line 8351 | line 11738 | YES — first does `vec_push` flow + `vec_get`, second does `__arena_push` + `__arena_get` |

Verified by `python -m pytest helixc/tests/test_codegen.py --collect-only -q` — only 1 node ID per name is collected.

This is qualitatively the same defect class flagged in earlier cycles (test-runner discovery / discrimination findings in cycle-83 and cycle-80). Pre-fix risk: the arena-based variant of `test_stdlib_vec_eq` (line 8331) — which would have exercised the 3-arg `vec_eq(a, b, n)` signature — has been silently disabled since the second `def` was added. If the 3-arg signature was ever the public surface, it has no regression coverage.

Recommended next step (out of scope for read-only audit): rename or delete the shadowed first-defined bodies, and add a lightweight collection guard (e.g. an `ast`-based pre-test that asserts no duplicate top-level `def` names per file in `helixc/tests/`).

### Out-of-scope-but-noted (cross-file same-name; informational only, NOT a finding)

10 cross-file same-name collisions exist (e.g. `test_function_call` in both `test_codegen.py` and `test_typecheck.py`). pytest disambiguates by file path so these are not bugs; flagging them here only as context for the project-wide naming hygiene picture.

---

## Areas reviewed clean

- **`check.py` flag dispatch** — `-W<name>[=<val>]` (deprecated/ad policies), `-O0..-O3` with the documented `-O3 == -O2` aliasing note explicit at lines 670–678, `-l` both separate (`-l m`) and attached (`-lm`) forms, `--doc` short-circuit before `parse()` at line 370 (correct — doc extraction is lexical). `_KNOWN_LONG_FLAGS` matches the help-text flag list at lines 19–47. `--check-only` early-exit at line 610 is documented as intentional (cycle-26 audit-R C25-1 note).
- **Cycle-86 follow-on (`const_fold.py` lines 488–520, `test_const_fold.py` lines 431–483)** — `_INT_BITS` table is the single source of truth and the SHL/SHR paths both consume it via the same `_INT_BITS.get(name, 64)` idiom. Error messages updated from `[0, 63]` to `[0, {bits})` consistently. Regression tests cover both the new failure case (i32 SHL=32) and boundary preservation (i64 SHL=63 still folds). No stale comments referencing the pre-fix `[0, 63]` literal survive elsewhere in `const_fold.py`.
- **`test_cli.py`** — 38 tests; CLI surface coverage is broad (parse_args matrix, doc extraction, main exit codes, AD-warning drain across default/check-only/subprocess paths, FoldError diagnostic, effect-check via check.py). No duplicate names.

---

## Summary

PASS criterion is 0 findings at conf >= 75%. C88-1 stands at conf 95 (verified by pytest collection), so the audit returns **FAIL**.

**No edits were made.** This file is the only write.
