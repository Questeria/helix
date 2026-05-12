# Audit Stage 28.9 cycle 92 — Code review

Scope: HEAD `d04e65b`. Strict read-only. Narrow scope — review cycle-91 docstring-clarity fix-sweep; spot-check other one-line docstrings; accept cycle-85/88 PASS verdict on `helixc/check.py` CLI completeness. Prior C1–C91 findings + deferred-known list NOT re-flagged.

Criterion: 0 findings at confidence >= 75 % = PASS.

---

## Verdict: **FAIL** — 1 finding at conf >= 75 %.

---

## Findings

### C92-1 [conf 90, HIGH] — `test_stdlib_vec_first_legacy_api` docstring inverts the API claim

File: `helixc/tests/test_codegen.py:11184–11202`.

The cycle-91 extended docstring states:

> The `_legacy_api` suffix is a bookkeeping marker NOT an API-version claim — **both variants currently exercise the `__arena_push` shape**. Keep both bodies during the Phase-0 stdlib transition; merge or further differentiate in a follow-up cycle.

The legacy body at L11194–11199 does use `__arena_push(...)` + `vec_first(v, 2)`. But the canonical `test_stdlib_vec_first` at L12814–12828 uses **`vec_push(s, idx, val)` (3-arg form)**, not `__arena_push`. The two bodies therefore exercise **different** API shapes — the same divergence that vec_eq_legacy_api / vec_reverse_inplace_legacy_api docstrings correctly call out for their own pair.

Concretely the legacy docstring's "both variants currently exercise the `__arena_push` shape" sentence is false; the sister `test_stdlib_vec_last_legacy_api` docstring inherits the error by reference ("See test_stdlib_vec_first_legacy_api above for full rationale"), so the bug propagates to L11205–11210 too.

Impact: the docstring tells a future reader that the bodies are redundant and one can be deleted — exactly the re-collapse hazard cycle-89 introduced the rename to prevent, and exactly what C90-1 said the extended docstrings must guard against. A reader trusting the docstring could re-collapse the pair and silently lose the `vec_push` 3-arg caller-form coverage.

Confidence rationale: read both bodies directly; `vec_push(s, 0, 42)` vs `__arena_push(42)` is unambiguous. The factual claim is wrong as written. HIGH (90).

---

## Notes (informational, not findings)

- vec_eq_legacy_api + vec_reverse_inplace_legacy_api docstring line references ("near line 13443", "near line 11738") drift to actual L13523 + L11818 — ~80–90 line gap. Soft-tolerance "near" wording is defensible; not flagged.
- 360 of 671 tests in `test_codegen.py` carry one-line docstrings. They describe self-contained simple semantics (e.g. `"max_pure([5,42,3]) = 42"`) without rename / shadow / API-shape hazards, so they are not in-scope for mandatory expansion. The cycle-89 / 91 thread is specific to the 4 `_legacy_api` deduped-rename pairs.
- `helixc/check.py` CLI completeness — cycle-85 / 88 PASS verdict accepted; no new finding.

No edits performed in this audit.
