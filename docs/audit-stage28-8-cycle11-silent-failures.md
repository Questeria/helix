# Stage 28.8 Cycle 11 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: c2e36d4 (read-only audit). **No new commits since
cycle 10** — `git log c2e36d4..HEAD` returns empty.
Cycle-11 re-audits the identical production-code surface that
cycle 10 found clean, to verify stability of the clean verdict
under re-audit with fresh eyes.
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically scanned for any silent-
failure window NOT counted in cycles 1-10 (new finding) and
deliberately did NOT re-flag findings already documented as
carryovers (audit-C4-1 CRITICAL, audit-C4-4 HIGH, audit-C4-8
LOW, C5-10 LOW, monomorphize_safe docstring drift, D-vs-Quote
diagnostic text, C7-1 test-coverage gap).
**Trigger**: pre-Stage-29 audit gate — Cycle 11 of 5+ clean-
cycle gate. Cycle 10 was the FIRST clean cycle of the gate
(counter: 1/5). Cycle 11 is the re-stability check on the same
HEAD.
**Strict criterion** (per user directive 2026-05-10, restated
for cycle 11): cycle counts CLEAN only when **zero new findings
of ANY severity** (CRITICAL/HIGH/MEDIUM/LOW). Findings already
documented in cycles 1-10 (and noted as deferred / open
carryovers) are NOT re-flagged unless they CHANGED since cycle
10 — which they did not, because no commit has landed since
c2e36d4.

**Method**:
1. Confirmed `git log c2e36d4..HEAD` is empty — HEAD is
   unchanged from cycle 10. Working tree contains only
   untracked audit doc files (`docs/audit-stage28-8-cycle10-
   *.md`, `docs/audit-stage28-8-cycle8-type-design-rev.md`)
   plus a modified `docs/audit-stage28-8-cycle8-silent-
   failures.md` whose modification is doc-only. No source-tree
   change.
2. Read the cycle-10 silent-failures audit verdict (0 new
   findings, CLEAN). Read its "Carryovers NOT re-flagged"
   ledger to enumerate exactly which findings must be excluded
   from the cycle-11 lens: audit-C4-1 (CRITICAL), audit-C4-4
   (HIGH), audit-C4-8 (LOW), monomorphize_safe docstring drift
   (housekeeping), D-vs-Quote diagnostic text (housekeeping),
   C7-1 test-coverage gap (carryover), C5-10 lower_ast fallback
   patterns (cycle-5 LOW, deferred and never closed — confirmed
   by `git log` on `lower_ast.py`: last touch was 134df9b
   "Audit 28.8 cycle 2 C2-2", well before cycle 5; the lines
   2113-2117 quote-handle structural_hash fallback have not
   changed since cycle 5 flagged them).
3. Walked the full `except` ledger in production code (excluding
   `helixc/tests/*`):
   - `helixc/check.py:299` `except (FileNotFoundError,
     PermissionError, IsADirectoryError, NotADirectoryError)`:
     audited in cycle 5 (C4-7 close), narrowed correctly.
   - `helixc/check.py:303` `except UnicodeDecodeError`: audited
     in cycle 5 (C4-6 close).
   - `helixc/check.py:306` `except Exception` (broad arm):
     audited in cycles 5/8 — the pre-cycle-9 ImportError arm was
     correctly DROPPED in cycle 9 so internal compiler bugs land
     here with rc=1 + "compiler bug" tag. Still correct.
   - `helixc/check.py:332` `except Exception as drain_e`
     (finally drain-failure suppressor): audited in cycle 5
     (C4-6 wrap), correctly emits a warning instead of masking.
   - `helixc/check.py:618`, `:649`, `:663` `except Exception`
     (backend-call wraps for --emit-asm, --emit-ptx, -o):
     audited in cycle 1 / Audit A9, correctly emit "internal
     error" + "compiler bug" + rc=1.
   - `helixc/frontend/autodiff.py:155` `except (TypeError,
     ValueError, AttributeError)`: narrowed in cycle 2 (C2-1
     deferred observation #20), emits AD warning on hash
     failure. No regression.
   - `helixc/frontend/autodiff.py:1012` `except (OverflowError,
     ZeroDivisionError, ValueError, TypeError)`: narrowed in
     cycle 2 (C2-1 deferred observation #19). No regression.
   - `helixc/frontend/autotune.py:80` `except ValueError`:
     narrowed in cycle 2. No regression.
   - `helixc/frontend/deprecated_pass.py:132` `except TypeError`:
     narrowed in cycle 2. No regression.
   - `helixc/frontend/grad_pass.py:641` `except (AttributeError,
     TypeError)`: pre-existing, type-narrow, fine.
   - `helixc/frontend/lexer.py:401` `except ValueError`:
     pre-existing, type-narrow, fine.
   - `helixc/frontend/monomorphize.py:203` `except ValueError`:
     pre-existing, type-narrow, fine.
   - `helixc/frontend/monomorphize.py:759` `except
     ShapeFoldError`: pre-existing, custom-type narrow, fine.
   - `helixc/frontend/panic_pass.py:97` `except TypeError`:
     narrowed in cycle 2. Fine.
   - `helixc/frontend/parser.py:375` `except ValueError`: type-
     narrow integer literal parse, fine.
   - `helixc/frontend/struct_mono.py:448,454` `except
     ShapeFoldError`, `except ValueError`: pre-existing, narrow.
   - `helixc/frontend/pytree.py:295` `except ValueError`: narrow.
   - `helixc/frontend/typecheck.py:415,423,636` narrow excepts
     for TypeError_/ValueError: pre-existing, narrow.
   - `helixc/frontend/unsafe_pass.py:93` `except TypeError`:
     narrowed in cycle 2. Fine.
   - `helixc/frontend/diagnostics.py:76` `except Exception`
     (isatty fallback to False): pre-existing, defensive — when
     `stream.isatty()` raises, no-color is the correct fail-
     safe (a pure UX preference fallback with no semantic loss).
     Never flagged as a finding in cycles 1-10. Re-confirmed
     non-finding for cycle 11.
   - `helixc/ir/lower_ast.py:282,2066` `except ValueError`:
     pre-existing, narrow.
   - `helixc/ir/lower_ast.py:2115` `except Exception` (quote-
     handle structural_hash → `_pretty` fallback): **C5-10 LOW
     carryover from cycle 5, deferred and never closed**. Not
     re-flagged for cycle 11 (no change since cycle 5; per the
     user's strict re-flag rule).
   - `helixc/ir/passes/const_fold.py:250,324,349,401` `except
     Exception` (arithmetic fold returns `None` on failure to
     fall back to runtime evaluation): defensive fail-open-to-
     runtime — the correct semantic for a fold pass is "leave
     the op alone if folding fails". The runtime semantics are
     unchanged. Not a silent failure (the op continues into
     codegen and the runtime path produces correct results).
     Never flagged in cycles 1-10. Re-confirmed non-finding.
4. Walked `git show c2e36d4` once more to verify the cycle-10
   commit truly added zero production code:
   - `docs/audit-stage28-8-cycle8-codereview-rev.md` (+559)
   - `docs/audit-stage28-8-cycle9-codereview.md` (+519)
   - `docs/audit-stage28-8-cycle9-silent-failures.md` (+627)
   - `docs/audit-stage28-8-cycle9-type-design.md` (+428)
   - `helixc/tests/test_typecheck.py` (+65, test-only)
   - **Zero production-code (.py outside tests, .hx) change.**
5. Spot-checked `getattr(it, "is_kernel", False)` at
   `check.py:641` for `--emit-ptx`: a soft attribute check
   defaults to False on FnDecls that don't carry the attr. This
   is acceptable because (a) the field is always present on
   FnDecl post-parse (defaults to False at construction time
   per `ast_nodes.py`) and (b) the fallback `False` is the
   right semantic (a non-kernel fn doesn't go into the PTX
   kernel-count). Not a silent failure.
6. Scanned all `except: pass` patterns: every occurrence is in
   `helixc/tests/*` (test_lexer, test_parser, test_codegen,
   test_reflection, test_select_codegen) — test infrastructure,
   not production code. Production code has **zero**
   `except: pass` patterns.
7. Confirmed no production code has try/finally suppression of
   the outer exception: the cycle-3 C3-3 `try/finally` in
   `check.main` correctly captures the inner rc and continues
   to drain; exceptions propagate normally up to the typed
   except arms.
8. Verified `check.py:_emit_env_error` is invariant under re-
   reading at HEAD: the helper still strips a single `helixc:`
   prefix on the message-shape the strict-stdlib raise produces
   (`parser.py:1587` raises `FileNotFoundError("helixc: stdlib
   file missing: <path>")`). End-to-end probe at HEAD: a fake
   `check.typecheck` raising `FileNotFoundError("helixc: x")`
   produces stderr `"helixc: x\n"` (count 1). No regression.
9. Ran `pytest helixc/tests/test_typecheck.py` at HEAD: 111/111
   pass. Ran `pytest helixc/tests/test_cli.py`: 38/38 pass.
   Identical to cycle-10 verdict.

**Result**: **0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW)** — Cycle 11 is **CLEAN** for the silent-failure audit
lens. Re-audit confirms cycle-10 clean verdict is stable:
- No new commit has landed since cycle 10, so by construction
  the production-code surface is identical and any clean
  verdict at cycle 10 must hold at cycle 11.
- The fresh re-walk did not surface any silent-failure window
  that cycle 10 might have overlooked. Every `except` arm in
  production code is either type-narrow (cycle 2 narrowing),
  a defensive fail-safe with correct semantic (const_fold,
  diagnostics isatty), a previously-flagged carryover (C5-10
  lower_ast.py:2115), or a correctly-emitting compiler-bug arm
  (check.py:306/618/649/663).
- All carryovers from cycles 1-10 are excluded from the
  re-flag (per user directive, restated above).

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

## Re-audit verification on c2e36d4 (same state as cycle 10)

The cycle-10 fix-sweep landed as a single commit (c2e36d4)
covering C9-1 plus persisted prior-cycle audit docs. No
subsequent commit has landed. The cycle-11 re-audit therefore
covers the IDENTICAL production-code surface.

| Re-audit pass | Cycle-10 verdict | Cycle-11 re-audit verdict | Stability |
|---|---|---|---|
| `_emit_env_error` strip helper (check.py:246-255) | clean | clean (re-probed end-to-end at HEAD) | stable |
| Outer-except topology (check.py:284-318) | clean | clean (ImportError correctly lands in broad arm; rc=1; "compiler bug" hint present) | stable |
| Finally drain-failure suppressor (check.py:319-337) | clean | clean (emits warning, doesn't mask) | stable |
| Backend-call wraps (check.py:618,649,663) | clean | clean (Audit A9 pattern preserved) | stable |
| AD-warning narrowed excepts (autodiff.py:155,1012) | clean | clean (cycle 2 narrowing preserved) | stable |
| const_fold defensive folds (const_fold.py:250,324,349,401) | clean | clean (fail-open-to-runtime — correct semantic) | stable |
| Quote-handle structural_hash fallback (lower_ast.py:2115) | C5-10 LOW carryover, NOT re-flagged | C5-10 LOW carryover, NOT re-flagged | stable carryover |
| diagnostics isatty fallback (diagnostics.py:76) | non-finding | non-finding (defensive UX fallback, correct semantic) | stable |
| `getattr(it, "is_kernel", False)` (check.py:641) | non-finding | non-finding (FnDecl always carries field; default False is correct semantic) | stable |
| No `except: pass` in production | clean | clean (all `except: pass` are in `helixc/tests/`) | stable |

### Specific items re-checked clean in cycle 11

- **No new commits → no new code surface**: `git log
  c2e36d4..HEAD` returns empty. By construction the cycle-10
  clean verdict propagates to cycle 11 unless I find a window
  cycle 10 overlooked. Fresh re-walk found no such window.
- **`_emit_env_error` strip on triple-prefix edge case**: still
  no callee emits triple-prefixed messages (`parser.py:1587` is
  the only callee that pre-prefixes, and it emits exactly one
  prefix). Not a finding (matches cycle 10's deferred-not-
  finding determination).
- **`_emit_env_error` uppercase-prefix edge case**: still no
  callee emits uppercase-prefixed messages. Not a finding.
- **Soft attribute access via `getattr(it, "is_kernel", False)`
  at check.py:641**: cross-checked `ast_nodes.py` — `FnDecl.
  is_kernel` is a field with default `False` (set at
  construction). The `getattr` with `False` default is
  effectively a redundant guard, not a silent failure (the
  field is always present and the soft-default matches the
  field's own default). Not a finding.
- **Stdout/stderr isolation for AD-warning drain
  (check.py:236-243)**: the drain prints the count line to
  stdout (`print(f"   ad:        {n} {label}(s)")`) and each
  individual warning to stderr. This is the cycle-2 C2-1
  topology, preserved. Not a finding.
- **Drain-failure warning at check.py:333-337**: emits to
  stderr with `helixc: warning:` prefix. The warning shape is
  intentionally lowercase-`warning` (not an error), since the
  drain failure is a secondary effect that should not mask the
  primary failure already in `rc`. Cycle-5 design choice, no
  regression. Not a finding.
- **The `if a_holder:` branch at check.py:325**: drain only
  runs when CliArgs were parsed successfully. If `parse_args`
  raised (impossible in current code path because it returns
  `(args, errors)` tuple — no raise), or if `_main_inner`
  raised BEFORE pushing to `a_holder`, the `else` branch
  silently calls `_drain_ad_init()` for state hygiene with no
  user feedback. **This is intentional** — if no CliArgs
  succeeded, there's no `a.warnings["ad"]` policy to consult,
  so any accumulated `_DIFF_WARNINGS` would belong to a
  caller-side compile (not the current attempted invocation).
  Draining quietly is the correct semantic. Not a finding.

### Cross-stage interactions re-checked (cycle 11)

- **`_emit_env_error` + cycle-9 broad-Exception arm
  interaction**: re-probed with a fake `check.typecheck`
  raising `ImportError("cannot import X")`. Result: stderr is
  `"helixc: internal error: ImportError: cannot import X\n
  helixc: this is a compiler bug — please file an issue.\n"`
  (two `helixc:` prefixes — but each on a separate line, as
  intended — first for the error itself, second for the file-
  an-issue hint). Not a double-prefix bug (the helper
  `_emit_env_error` is NOT called from this arm; the broad arm
  prints directly via `print(..., file=sys.stderr)`). Not a
  finding.
- **`_emit_env_error` + finally drain interaction**: re-probed
  with a fake `check.typecheck` raising `FileNotFoundError("x")`
  AFTER `a_holder.append(a)` was executed. Result: the env-
  error arm prints `"helixc: x\n"`, the finally branch then
  runs `_drain_ad_warnings(a_holder[0])` cleanly (no warnings
  to drain because typecheck never completed), drain_rc=0,
  rc=2 preserved. Not a finding.
- **`_emit_env_error` + drain-failure interaction**: re-probed
  with a fake `take_diff_warnings` raising RuntimeError in
  the finally. Result: the outer arm has already set rc=2, the
  drain raises, the inner `except Exception as drain_e` catches
  it and emits `"helixc: warning: AD-warning drain failed:
  RuntimeError: <msg>\n"`. The original env-error stderr is
  preserved; the warning is a separate stderr line. Not a
  finding.

### Carryover findings status (cycles 1-10) — unchanged

The cycle-11 re-audit DID NOT close any carryover (no new
commits), and DID NOT introduce any new finding. The carryover
ledger is identical to cycle 10's closing snapshot.

| Carryover | Severity | Cycle-11 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — not addressed; same deferral rationale (parse-time constant folding or pre-closure-capture typecheck pass needed). Cycle 11 did not address. |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 deferred (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| C5-10 (lower_ast.py:2113-2117 quote-handle fallback + Cast None→0 + Field None-return) | LOW | **still open** — not addressed; not re-flagged for cycle 11 per the user's strict re-flag rule. Re-flag would block the clean-counter without adding new information. |
| monomorphize_safe docstring drift (cycle-6 deferred) | (not a finding) | **still open** — docstring still suggests callers MAY ignore diags; only caller now aborts |
| D-vs-Quote diagnostic text (cycle-7 deferred) | (not a finding) | **still open** — Quote-wrapped case still emits "(one side D-wrapped, other bare)" |
| C7-1 test-coverage gap (cycle-8/9/10 deferred) | (not a finding) | **still open** — cycle 11 did not add the `_compatible(TyMemTier, TyVar) is False` regression tests either. Not re-flagged here (carryover; did not CHANGE in cycle 11). |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | **CLOSED by cycle 9** (regression test added cycle 10) |
| C8-2 (cycle-8 LOW) | LOW | **CLOSED by cycle 9** (regression test added cycle 10) |
| C9-1 (cycle-9 LOW) | LOW | **CLOSED by cycle 10** |
| C10 silent-failure | n/a | **0 new findings (CLEAN)** |

These are NOT re-flagged as new cycle-11 findings per the
user directive (already documented in cycles 1-10, did not
CHANGE in cycle 11 — and indeed could not have changed because
no commit landed). They remain in the open-findings ledger and
are out-of-scope for this audit's strict-clean determination.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-12 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 11 did not address (read-only re-audit
  cycle by definition — no fixes intended). **STILL THE
  HIGHEST-PRIORITY ITEM** for any future fix-sweep — the only
  remaining CRITICAL across the audit series. The clean-counter
  is at 1/5; once cycles 11-14 are all clean, the gate fires
  even with audit-C4-1 still open IF that's the user's
  intended interpretation of the gate. If the gate also
  requires CRITICAL=0 open (stricter interpretation), then
  audit-C4-1 must be closed before the gate fires.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH.
  Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks**: still open LOW
  (Pattern A: quote-handle structural_hash → _pretty fallback;
  Pattern B: Cast None inner → const_int(0); Pattern C: Field
  no-array-match returns None). Not addressed; not re-flagged.
- **monomorphize_safe docstring drift**: still open (cycle-6
  deferred). Docstring suggests callers MAY ignore diags; the
  only caller (x86_64.py) now aborts.
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred). Quote-wrapped case still emits "(one side
  D-wrapped, other bare)".
- **C7-1 test-coverage gap**: still open (cycle-8/9/10
  deferred). Cycle 11 also did not add the 4 `_compatible
  (TyMemTier, TyVar)` regression tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix edge
  cases**: still no callee triggers either. Not findings.

---

## Cycle 10 vs cycle 11 — clean-cycle counter check

Cycle 10 was the 1st clean cycle of the 5-clean-cycle gate
(counter: 1/5 after cycle 10). The user directive for cycle
11 explicitly instructs: "DO NOT re-flag prior-cycle findings
that were already counted as carryovers in cycle 10's
'Carryovers NOT re-flagged' list — they're stable since cycle-
10 and re-flagging them would block the counter without adding
new information."

The cycle-11 re-audit honors that directive:
- `audit-C4-1 CRITICAL`, `audit-C4-4 HIGH`, `audit-C4-8 LOW`:
  not re-flagged.
- `C5-10 LOW` (lower_ast.py:2113-2117 + 2093-2101 + 2079-2092):
  not re-flagged. (Note: this carryover was implicitly in the
  cycle-10 ledger via "Carryovers ... still open" but not
  enumerated explicitly. Adding it explicitly to the cycle-11
  ledger for completeness; not re-flagging as a new finding.)
- `monomorphize_safe docstring drift`, `D-vs-Quote diagnostic
  text`, `C7-1 test-coverage gap`: not re-flagged.

Cycle 11 produces **zero NEW findings of any severity**, so
the clean-cycle counter advances to **2/5** (cycle 10 = 1,
cycle 11 = 2) under the strict criterion — subject to the
parallel type-design + code-review audit lenses also being
clean for cycle 11.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 11 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW).**

---

## Cycle 11 status

**Cycle 11 IS CLEAN** for the silent-failure audit lens. Per
the strict criterion (zero findings of ANY severity), the
0-finding result satisfies the clean-cycle gate for this
audit lens.

### Stop-the-line determination: **NO**

Cycle 11 is clean — no stop required for this lens.

### Cycle 11 → NEW FINDINGS COUNT for the strict-clean gate: 0 (0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter advances to **2/5** for this audit lens (subject to the parallel type-design + code-review audit lenses also being clean for cycle 11).

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
- Cycle 11: 0 findings. ← here

Trend: 2 consecutive clean cycles. Cycle 11's clean verdict
is by construction (no new commits) plus a fresh-eyes re-walk
confirming cycle 10 did not overlook a window. The audit
series is stable at zero new findings.

### Estimated remaining open findings going into cycle 12

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 5 silent-failure: 4 new — 3 closed by cycle 6
  (C5-5, C5-6, C5-7, C5-8 MEDIUM and C5-9 LOW), 1 still open
  (C5-10 LOW, lower_ast.py fallbacks).
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new (C8-1 MEDIUM, C8-2 LOW) —
  both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10 silent-failure: 0 new.
- Cycle 11 silent-failure: 0 new. ← here
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 11).
- Cycle 11 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-11 new) + (deferred type-design partial) = **≥23
  open findings** going into cycle 12. (Net 0 delta from
  cycle 10: cycle 11 closed nothing, opened nothing.)

Recommend prioritizing in this order for the cycle-12 fix
batch (if user elects to land fixes between clean re-audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in cycles
   6-11; the carryover deadline approaches as the strict-
   clean gate accumulates clean cycles).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks).
4. **C7-1 test-coverage gap** (combinable with audit-C4-1
   if the fix touches typecheck.py).
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).

The "5 clean cycles before Phase 0 deprecation" goal requires
the strict criterion (zero findings of any severity) to be met
for 5 CONSECUTIVE cycles. Cycle 10 was the 1st; cycle 11 is
the 2nd. Three more clean cycles (12, 13, 14) and the gate
fires for this audit lens. The cycle-11 re-audit confirms the
production-code surface is stable: a re-audit on identical
HEAD with fresh eyes finds no overlooked silent-failure
window.
