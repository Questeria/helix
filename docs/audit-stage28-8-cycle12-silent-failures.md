# Stage 28.8 Cycle 12 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: df825ac (read-only audit). Commits since c2e36d4:
`git log --oneline c2e36d4..HEAD` →
- df825ac "Audit 28.8 cycle 11: persist 3 cycle-11 CLEAN audit docs"
- 9685c3a "Persist cycle-10 audit docs (silent-failures + type-design + codereview)"

Both commits are **DOC-ONLY**. `git diff --stat c2e36d4..HEAD`
shows changes exclusively under `docs/audit-stage28-8-cycle{10,11}-*.md`
(6 new doc files, 2603 lines of audit prose). **Zero
production-code (.py outside tests, .hx) delta.**

**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. The cycle-12 lens is identical to cycles
10/11: any silent-failure window NOT already counted in cycles
1-11 as a carryover. Documented carryovers (audit-C4-1 CRITICAL,
audit-C4-4 HIGH, audit-C4-8 LOW, C5-10 LOW, monomorphize_safe
docstring drift, D-vs-Quote diagnostic text, C7-1 test-coverage
gap) are NOT re-flagged per the user's strict re-flag rule (a
carryover is re-flagged only if it CHANGED since the prior
cycle — and none did, because no production code changed).

**Trigger**: pre-Stage-29 audit gate — Cycle 12 of the 5+
clean-cycle gate. Cycle 10 was the 1st clean cycle (counter
1/5), cycle 11 was the 2nd (counter 2/5). Cycle 12 is the
re-stability check at df825ac.

**Strict criterion** (per user directive 2026-05-10): cycle
counts CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW) at the confidence-≥80 reporting
threshold. Findings already in the carryover ledger are
explicitly excluded.

**Rotation for fresh-eyes coverage**: cycle 11 emphasized the
full `except` ledger walk + cross-stage check.py interactions.
Cycle 12 rotates the spot-check surface to:
- `helixc/ir/lower_ast.py` (try/finally scope-management
  pattern at :596 and :1800 — distinct from the :2115
  C5-10-carryover quote-handle fallback)
- `helixc/backend/x86_64.py` (the 3123-line largest backend;
  zero try/except, but defensive `attrs.get(..., default)`
  reads)
- `helixc/backend/ptx.py` + `helixc/backend/elf_dyn.py` (the
  two other backends; zero try/except)
- `helixc/ir/tile_ir.py` + `helixc/ir/tir.py` (IR container
  modules; zero raise / except)
- `helixc/frontend/parser.py:375` (integer-literal ValueError
  → re-raised as ParseError) and `frontend/typecheck.py:415,
  423, 636` (domain-typed TypeError_ collected as diagnostics
  + Optional[LinExpr] None-return for non-numeric size)
- `helixc/frontend/monomorphize.py:203` (size-name ValueError
  → leave expr unchanged for typecheck to flag) and `:759`
  (ShapeFoldError → diag list)

**Method**:
1. `git log --oneline c2e36d4..HEAD` listed two commits, both
   doc-only. `git diff --stat c2e36d4..HEAD` confirmed no .py
   outside docs and no .hx changes since cycle 10.
2. Read the cycle-11 silent-failures audit verdict (0 new
   findings, CLEAN; counter 2/5). Read its "Carryovers NOT
   re-flagged" ledger to enumerate exactly which findings are
   excluded from the cycle-12 lens.
3. Walked the full production `except` ledger again,
   cross-checking against the cycle-11 enumeration:
   - `helixc/check.py:299` `except (FileNotFoundError,
     PermissionError, IsADirectoryError, NotADirectoryError)`:
     audited in cycle 5 (C4-7 close), narrowed correctly.
   - `helixc/check.py:303` `except UnicodeDecodeError`:
     audited in cycle 5 (C4-6 close).
   - `helixc/check.py:306` `except Exception` (broad arm):
     audited in cycles 5/8/9 (ImportError arm correctly
     dropped in cycle 9 → broad arm catches internal bugs
     with rc=1 + "compiler bug" tag). Still correct.
   - `helixc/check.py:332` `except Exception as drain_e`
     (finally drain suppressor): audited in cycle 5 (C4-6
     wrap), emits a warning rather than masking.
   - `helixc/check.py:382` `except ParseError`, `:443` `except
     DuplicateMethodError`, `:676` `except OSError`:
     domain-typed narrow excepts that surface user-facing
     diagnostics with rc!=0. No silent failure.
   - `helixc/check.py:618,649,663` (backend-call wraps for
     --emit-asm / --emit-ptx / -o): Audit 28.8 A9 pattern,
     emits "internal error" + "compiler bug" + rc=1.
   - `helixc/frontend/autodiff.py:155` `except (TypeError,
     ValueError, AttributeError)`: cycle-2 narrowing (C2-1
     observation #20), emits AD warning on hash failure. No
     regression.
   - `helixc/frontend/autodiff.py:1012` `except (OverflowError,
     ZeroDivisionError, ValueError, TypeError)`: cycle-2
     narrowing (C2-1 observation #19). No regression.
   - `helixc/frontend/autotune.py:80` `except ValueError`:
     cycle-2 narrowing.
   - `helixc/frontend/deprecated_pass.py:132` `except
     TypeError`: cycle-2 narrowing.
   - `helixc/frontend/grad_pass.py:641` `except (AttributeError,
     TypeError)`: type-narrow, pre-existing, fine.
   - `helixc/frontend/lexer.py:401` `except ValueError`:
     type-narrow integer literal parse, fine.
   - `helixc/frontend/monomorphize.py:203` `except ValueError`:
     pre-existing, type-narrow. Re-checked: when `int(size_<N>)`
     fails, returns original `expr` so downstream typecheck
     flags the bad size literal. Not silent.
   - `helixc/frontend/monomorphize.py:759` `except
     ShapeFoldError`: domain-typed, returns `(0, [str(e)])`
     for the caller (x86_64.py) which aborts. Docstring drift
     (cycle-6 housekeeping carryover) NOT re-flagged.
   - `helixc/frontend/panic_pass.py:97` `except TypeError`:
     cycle-2 narrowing. Fine.
   - `helixc/frontend/parser.py:375` `except ValueError`:
     re-raises as `ParseError(f"bad integer literal {t.value!r}",
     t)` — user-visible diagnostic, not silent.
   - `helixc/frontend/struct_mono.py:448,454` `except
     ShapeFoldError`, `except ValueError`: pre-existing,
     narrow.
   - `helixc/frontend/pytree.py:295` `except ValueError`:
     narrow.
   - `helixc/frontend/typecheck.py:415,423`: `except
     TypeError_` (domain type) — appends to `self.errors`,
     surfaced as diagnostics. Not silent.
   - `helixc/frontend/typecheck.py:636`: `except ValueError`
     in `_size_type_to_lin` — function signature is
     `Optional[LinExpr]`, so returning `None` on `int(size_<N>)`
     failure matches the contract (caller handles None
     explicitly). Not silent.
   - `helixc/frontend/unsafe_pass.py:93` `except TypeError`:
     cycle-2 narrowing. Fine.
   - `helixc/frontend/diagnostics.py:76` `except Exception`
     (isatty fallback to False): defensive UX fallback (no-
     color is the right fail-safe when stream.isatty() raises).
     Pre-existing, never flagged in cycles 1-11. Re-confirmed
     non-finding for cycle 12.
   - `helixc/ir/lower_ast.py:282, 2066` `except ValueError`:
     pre-existing narrow.
   - `helixc/ir/lower_ast.py:596, 1800` `try: ... finally:
     self._pop_scope()`: scope-management pattern, exception
     propagates upward unchanged. Not silent. (New cycle-12
     spot-check; not flagged in any prior cycle. Confirmed
     correct.)
   - `helixc/ir/lower_ast.py:2115` `except Exception` (quote-
     handle structural_hash → `_pretty` fallback): **C5-10 LOW
     carryover**, deferred and never closed. Not re-flagged
     for cycle 12 (no change since cycle 5).
   - `helixc/ir/passes/const_fold.py:250, 324, 349, 401`
     `except Exception` (arithmetic fold returns `None` /
     `False` on failure to fall back to runtime): defensive
     fail-open-to-runtime — the correct semantic for a fold
     pass is "leave the op alone if folding fails". The
     runtime semantics are unchanged. Not silent. Stable
     non-finding.
4. Verified `helixc/backend/{x86_64.py, ptx.py, elf_dyn.py}`:
   zero `try` / `except`. The three backends are exception-
   transparent (any internal Python error propagates up to
   `check.py:618/649/663`'s broad-Exception wrap which emits
   "internal error" + "compiler bug" + rc=1). Not silent.
5. Verified `helixc/ir/{tile_ir.py, tir.py}`: zero `try` /
   `except` / `raise`. Pure IR containers. Not applicable to
   the silent-failure lens.
6. Spot-checked `helixc/backend/x86_64.py` `attrs.get(<key>,
   <default>)` defaults (lines 871, 872, 880, 1596, 1659,
   1702, 1897, 1929, 2453, 2479, 2509, 2510, 2546, 2606,
   2639): defaults are obvious-garbage values (`"?"`, `""`,
   `0`, `28501` trap-id sentinel) that would surface in the
   emitted asm as clearly-wrong output — not silent failures.
   IR-construction-side guarantees set these attrs (e.g.
   `lower_ast.py` constructs CALL with `target`, PRINT_STR
   with `text`, TRAP with `trap_id`). The `.get` defaults are
   defensive-only. Same pattern as cycle-11 `getattr(it,
   "is_kernel", False)` at `check.py:641` (re-confirmed
   non-finding). Never flagged in cycles 1-11. Re-confirmed
   non-finding for cycle 12.
7. Re-walked all `except: pass` patterns. Every occurrence is
   in `helixc/tests/*` (test_lexer, test_parser, test_codegen,
   test_reflection, test_select_codegen, test_match). Test
   infrastructure, not production. Production code has **zero**
   `except: pass`.
8. Confirmed no production code adds try/finally suppression
   of the outer exception since cycle 11. The cycle-3 C3-3
   `try/finally` in `check.main` correctly captures the inner
   rc and propagates the outer exception type to the
   appropriate typed-except arm.
9. Verified `check.py:_emit_env_error` is invariant under
   re-reading at HEAD (df825ac is doc-only since c2e36d4,
   which is identical to the cycle-11 HEAD).
10. Re-ran `pytest helixc/tests/test_typecheck.py
    helixc/tests/test_cli.py` at HEAD: **149/149 pass**
    (test_typecheck.py 111/111 + test_cli.py 38/38).
    Identical to cycle-11 verdict.

**Result**: **0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW)** — Cycle 12 is **CLEAN** for the silent-failure
audit lens. Re-audit confirms cycle-10 and cycle-11 clean
verdicts are stable across three consecutive read-only re-
walks:
- No production-code commit has landed since c2e36d4 (the
  cycle-10 commit). By construction, the production-code
  surface is identical and any clean verdict at cycle 10
  must hold at cycle 12.
- The fresh re-walk for cycle 12 rotated the spot-check
  surface (lower_ast.py try/finally scope-management
  pattern at :596 + :1800, backend/x86_64.py defensive
  `attrs.get`, backend/ptx.py + elf_dyn.py zero-except,
  ir/tile_ir.py + tir.py zero-raise) and did not surface
  any silent-failure window that cycles 10-11 might have
  overlooked.
- Every `except` arm in production code is either type-
  narrow (cycle 2 narrowing), a defensive fail-safe with
  correct semantic (const_fold, diagnostics isatty, lower_ast
  scope try/finally, backend attrs.get), a previously-
  flagged carryover (C5-10 lower_ast.py:2115), or a
  correctly-emitting compiler-bug arm (check.py:306/618/649/
  663).
- All carryovers from cycles 1-11 are excluded from the
  re-flag (per user directive).

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

(none)

---

## MEDIUM FINDINGS

(none)

---

## LOW FINDINGS

(none)

---

## Re-audit verification on df825ac (production surface identical to c2e36d4)

The cycle-10 fix-sweep landed as a single commit (c2e36d4)
covering C9-1 plus persisted prior-cycle audit docs. The two
subsequent commits (9685c3a, df825ac) are doc-only persistence
of cycle-10/11 audit deliverables. The cycle-12 re-audit
therefore covers the IDENTICAL production-code surface as
cycles 10 and 11.

| Re-audit pass | Cycle-10 | Cycle-11 | Cycle-12 | Stability |
|---|---|---|---|---|
| `_emit_env_error` strip helper (check.py:246-255) | clean | clean | clean | stable |
| Outer-except topology (check.py:284-318) | clean | clean | clean | stable |
| Finally drain-failure suppressor (check.py:319-337) | clean | clean | clean | stable |
| Backend-call wraps (check.py:618,649,663) | clean | clean | clean | stable |
| AD-warning narrowed excepts (autodiff.py:155,1012) | clean | clean | clean | stable |
| const_fold defensive folds (const_fold.py:250,324,349,401) | clean | clean | clean | stable |
| Quote-handle fallback (lower_ast.py:2115) | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| diagnostics isatty fallback (diagnostics.py:76) | non-finding | non-finding | non-finding | stable |
| `getattr(it, "is_kernel", False)` (check.py:641) | non-finding | non-finding | non-finding | stable |
| lower_ast.py try/finally scope at :596, :1800 | (not enumerated) | (not enumerated) | **cycle-12 fresh spot-check: clean** (try/finally for scope hygiene, exception propagates) | stable |
| backend/x86_64.py attrs.get defaults | (not enumerated) | (not enumerated) | **cycle-12 fresh spot-check: clean** (defaults are obvious-garbage that surfaces in asm; IR-side guarantees set the attrs) | stable |
| backend/ptx.py, elf_dyn.py zero-except | (not enumerated) | (not enumerated) | **cycle-12 fresh spot-check: clean** (no try/except; exceptions propagate to check.py:618/649/663 wraps) | stable |
| ir/tile_ir.py, tir.py zero-raise | (not enumerated) | (not enumerated) | **cycle-12 fresh spot-check: not applicable** (pure containers, no exception sites) | n/a |
| frontend/parser.py:375 ValueError → ParseError re-raise | clean | clean | clean (fresh re-read) | stable |
| frontend/typecheck.py:415,423 TypeError_ → diag append | clean | clean | clean (fresh re-read) | stable |
| frontend/typecheck.py:636 ValueError → Optional None | clean | clean | clean (fresh re-read, contract-correct) | stable |
| frontend/monomorphize.py:203 ValueError → return expr | clean | clean | clean (fresh re-read, leaves expr for typecheck) | stable |
| frontend/monomorphize.py:759 ShapeFoldError → diag list | clean | clean | clean (docstring drift carryover, not re-flagged) | stable |
| No `except: pass` in production | clean | clean | clean (all `except: pass` are in `helixc/tests/`) | stable |

### Specific items re-checked clean in cycle 12

- **No new production commits → no new code surface**: `git
  diff --stat c2e36d4..HEAD` is doc-only (2603 doc-line
  additions, 0 .py outside docs, 0 .hx). By construction the
  cycle-11 clean verdict propagates to cycle 12 unless the
  fresh re-walk finds an overlooked window. None found.
- **`lower_ast.py:596` `_lower_block` try/finally scope**:
  the `try` body lowers statements and the optional final
  expression; the `finally` calls `self._pop_scope()`. Any
  exception raised by `_lower_stmt` or `_lower_expr`
  propagates up after scope cleanup — not silenced. Cycle-12
  fresh spot-check, not previously enumerated in the cycle-11
  table but logically covered by "every `try` in production
  has a matching narrow `except` or is a try/finally for
  resource cleanup". Confirmed non-finding.
- **`lower_ast.py:1800` for-loop body scope try/finally**:
  identical pattern (push_scope / body / pop_scope in
  finally). Exceptions propagate. Confirmed non-finding.
- **`backend/x86_64.py` `attrs.get(key, default)` reads**: 15
  occurrences, all using obvious-garbage defaults (`"?"`,
  `""`, `0`, `28501` for trap-id sentinel). The IR builder
  in `lower_ast.py` constructs the operative ops with these
  attrs populated; the `.get` defaults are defensive-only,
  not the active code path. If the IR builder ever
  regressed and forgot to set one of these attrs, the
  emitted asm would contain visibly-wrong values (e.g., a
  CALL to `"?"`, a PRINT_STR of `""`, a TRAP with id 28501).
  These are NOT silent failures — they'd surface in test
  output or at runtime. Cycle-12 fresh spot-check, not
  previously enumerated. Confirmed non-finding.
- **`backend/ptx.py` + `backend/elf_dyn.py` zero-except**:
  any internal error propagates up through the backend
  function call to `check.py:649` (PTX) / `check.py:663`
  (object emit), which wrap with `except Exception as e`
  and emit "internal error" + "compiler bug" + rc=1. Cycle-12
  fresh spot-check; confirmed the cycle-11 backend-wrap
  topology covers both backends. Confirmed non-finding.
- **`ir/tile_ir.py` + `ir/tir.py` zero-raise**: the IR
  containers do not raise (they're builder primitives /
  dataclass-style modules). The silent-failure lens does
  not apply to modules that don't raise; their callers
  (lower_ast, codegen, passes) handle errors. Confirmed
  not-applicable.
- **Stdlib + bootstrap .hx unchanged**: `git diff
  c2e36d4..HEAD -- helixc/stdlib helixc/bootstrap` is
  empty. The .hx audit surface is unchanged from cycle-11
  clean. Cycle-12 inherits the clean verdict by construction.

### Cross-stage interactions re-checked (cycle 12)

- **check.py outer-except + backend-wrap interaction**: if
  `helixc.backend.x86_64.compile(prog)` raises `KeyError`
  during emit (e.g., from a missing op attr that the
  defensive `.get` default doesn't cover — though all
  current `.get` calls DO have defaults), the exception
  propagates up to `check.py:663`'s broad-Exception arm
  which emits "internal error: KeyError: ..." + "compiler
  bug" + rc=1. Cycle-11 verified this; cycle 12 re-verifies
  by re-reading the backend → `check.py:618-665` topology
  and confirming no intermediate `except` swallows it. Not
  a finding.
- **monomorphize_safe → x86_64.py abort chain**:
  `monomorphize_safe` returns `(0, ["msg"])` on
  ShapeFoldError; x86_64.py's caller checks `diags` and
  aborts. The cycle-6 housekeeping carryover (docstring
  says callers "MAY ignore diags" but the only caller now
  aborts) does not introduce a silent failure — the
  diags-bypass code path is dead. Not a finding.
- **typecheck.py:_size_type_to_lin → Presburger consumer**:
  returns `Optional[LinExpr]`; callers (e.g.
  `_size_expr_to_lin`, shape-constraint solver entry) check
  `if lin is None: ...` and either skip the constraint or
  emit a typecheck diagnostic. The `None` return on
  non-numeric size is contract-correct, not silent.

### Carryover findings status (cycles 1-11) — unchanged

The cycle-12 re-audit closed nothing (read-only by design)
and introduced no new finding. The carryover ledger is
identical to cycle 11's closing snapshot.

| Carryover | Severity | Cycle-12 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — not addressed. Highest-priority unaddressed-CRITICAL. |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| C5-10 (lower_ast.py:2113-2117 + 2079-2092 + 2093-2101) | LOW | **still open** — not addressed; not re-flagged per the user's strict re-flag rule |
| monomorphize_safe docstring drift | (housekeeping) | **still open** — docstring still suggests callers MAY ignore diags |
| D-vs-Quote diagnostic text | (housekeeping) | **still open** — Quote-wrapped case still emits "(one side D-wrapped, other bare)" |
| C7-1 test-coverage gap | (housekeeping) | **still open** — `_compatible(TyMemTier, TyVar) is False` regression tests not added |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | CLOSED by cycle 9 |
| C8-2 (cycle-8 LOW) | LOW | CLOSED by cycle 9 |
| C9-1 (cycle-9 LOW) | LOW | CLOSED by cycle 10 |
| C10 silent-failure | n/a | **0 new findings (CLEAN)** |
| C11 silent-failure | n/a | **0 new findings (CLEAN)** |

These are NOT re-flagged as new cycle-12 findings per the
user directive (already documented in cycles 1-11, did not
CHANGE in cycle 12 — and indeed could not have changed
because no production commit landed). They remain in the
open-findings ledger and are out-of-scope for this audit's
strict-clean determination.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-13 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 12 did not address (read-only
  re-audit by definition). **STILL THE HIGHEST-PRIORITY
  ITEM** for any future fix-sweep — the only remaining
  CRITICAL across the audit series. As the clean-counter
  accumulates (now 3/5 if cycle 12's clean verdict holds
  across all three audit lenses), the question of whether
  the Stage-29 gate requires CRITICAL=0-open (stricter
  interpretation) or merely 5-consecutive-clean (lenient
  interpretation) becomes load-bearing. The cycle-11
  recommendation stands: prioritize audit-C4-1 in the next
  fix-sweep regardless.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH.
  Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks**: still open LOW
  (Pattern A: quote-handle structural_hash → _pretty
  fallback at :2115; Pattern B: Cast None inner → const_int(0);
  Pattern C: Field no-array-match returns None). Not
  addressed; not re-flagged.
- **monomorphize_safe docstring drift**: still open (cycle-6
  deferred).
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred).
- **C7-1 test-coverage gap**: still open. Cycle 12 also did
  not add the 4 `_compatible(TyMemTier, TyVar)` regression
  tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix edge
  cases**: still no callee triggers either. Not findings.

---

## Cycle 11 vs cycle 12 — clean-cycle counter check

Cycle 10 = 1st clean (counter 1/5). Cycle 11 = 2nd clean
(counter 2/5). The user directive for cycle 12 explicitly
instructs: re-audit the same scope and confirm nothing has
regressed; do not re-flag prior-cycle carryovers.

The cycle-12 re-audit honors that directive:
- `audit-C4-1 CRITICAL`, `audit-C4-4 HIGH`, `audit-C4-8 LOW`:
  not re-flagged.
- `C5-10 LOW` (lower_ast.py:2113-2117 + 2093-2101 +
  2079-2092): not re-flagged.
- `monomorphize_safe docstring drift`, `D-vs-Quote
  diagnostic text`, `C7-1 test-coverage gap`: not re-flagged.

Cycle 12 produces **zero NEW findings of any severity**, so
the clean-cycle counter advances to **3/5** (cycle 10 = 1,
cycle 11 = 2, cycle 12 = 3) under the strict criterion —
subject to the parallel type-design + code-review audit
lenses also being clean for cycle 12.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 12 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW).**

---

## Cycle 12 status

**Cycle 12 IS CLEAN** for the silent-failure audit lens. Per
the strict criterion (zero findings of ANY severity), the
0-finding result satisfies the clean-cycle gate for this
audit lens.

### Stop-the-line determination: **NO**

Cycle 12 is clean — no stop required for this lens.

### Cycle 12 → NEW FINDINGS COUNT for the strict-clean gate: 0 (0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter advances to **3/5** for this audit lens (subject to the parallel type-design + code-review audit lenses also being clean for cycle 12).

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 8: 2 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW).
- Cycle 9: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 10: 0 findings.
- Cycle 11: 0 findings.
- Cycle 12: 0 findings. ← here

Trend: **3 consecutive clean cycles**. Cycle 12's clean
verdict is by construction (no new production commits since
c2e36d4) plus a fresh-eyes re-walk that rotated the spot-
check surface to lower_ast.py try/finally scope-management,
backend/x86_64.py defensive `attrs.get`, backend/ptx.py +
elf_dyn.py zero-except, and ir/tile_ir.py + tir.py zero-
raise — all confirmed non-findings. The audit series is
stable at zero new findings.

### Estimated remaining open findings going into cycle 13

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 5 silent-failure: 4 new — 3 closed by cycle 6
  (C5-5, C5-6, C5-7, C5-8 MEDIUM and C5-9 LOW), 1 still
  open (C5-10 LOW, lower_ast.py fallbacks).
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new (C8-1 MEDIUM, C8-2 LOW) —
  both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10 silent-failure: 0 new.
- Cycle 11 silent-failure: 0 new.
- Cycle 12 silent-failure: 0 new. ← here
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 13).
- Cycle 12 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-12 new) + (deferred type-design partial) = **≥23
  open findings** going into cycle 13. (Net 0 delta from
  cycles 10/11: cycle 12 closed nothing, opened nothing.)

Recommend prioritizing in this order for the cycle-13 fix
batch (if user elects to land fixes between clean re-audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in
   cycles 6-12; the carryover deadline approaches as the
   strict-clean gate accumulates clean cycles).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks).
4. **C7-1 test-coverage gap** (combinable with audit-C4-1
   if the fix touches typecheck.py).
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).

The "5 clean cycles before Phase 0 deprecation" goal
requires the strict criterion (zero findings of any
severity) to be met for 5 CONSECUTIVE cycles. Cycle 10 was
the 1st; cycle 11 the 2nd; cycle 12 the 3rd. Two more clean
cycles (13, 14) and the gate fires for this audit lens. The
cycle-12 re-audit confirms the production-code surface
remains stable: a re-audit on identical production HEAD
(only doc commits since c2e36d4) with rotated fresh-eyes
spot-checks finds no overlooked silent-failure window.

**Cycle 12 status: CLEAN**
**Counter status: 3/5** (cycles 10, 11, 12 all clean for
the silent-failure audit lens; subject to the parallel
type-design + code-review lenses also being clean for
cycle 12).
