# Stage 28.8 Pre-29 Audit Gate — Cycle 8, Audit C: Code Review (Revised at HEAD)

**Date**: 2026-05-11
**Commit (HEAD)**: 6968755 (read-only). Delta range
b8e047e..6968755 (cycle-7 fix-sweep through cycle-9 fix-sweep,
inclusive of cycle-5 cluster-fix carryover commits and the
cycle-8 G2 carve-out drop).
**Scope**: Re-audit of cycle-8 code-review, widened from the
prior pinned commit (5d1ca24) to the current HEAD. The prior
cycle-8 codereview doc (`audit-stage28-8-cycle8-codereview.md`,
persisted at 5d1ca24) covered only the G2 carve-out drop and the
C4-6 check.py classifier as it stood at 5d1ca24. Subsequent
commits added cycle-5 cluster fixes that landed AT or AFTER 5d1ca24
(C4-1 / C4-3 / C4-4 / C4-5 / C4-6 / C4-7 / C4-8 / F2..F10) and the
cycle-9 fix-sweep (which closed C8-1 + C8-2 against the
cycle-8 silent-failures findings against check.py's classifier).
This revised audit covers the full b8e047e..HEAD delta.

**Files reviewed at HEAD (6968755)**:

1. `helixc/bootstrap/parser.hx` — cycle-5 C4-1 / F1 (CRITICAL):
   FULL revert of cycle-3 D2's Call-RHS sentinel-12 arm. Both the
   `val_tag == 6` (cycle-4 broadening, reverted in cycle 6) and the
   `val_tag == 16` (cycle-3 D2 Call-only sentinel) arms are now
   gone. Verified that parser.hx no longer contains either gate;
   only literal-RHS arms remain (val_tag 0/27/31/34/35..41).
2. `helixc/frontend/monomorphize.py` lines 422-492 — cycle-5
   C4-4 / HIGH: `Monomorphizer.run` iteration order rework.
   Promoted-clone walk-set + non-generic-only top-level walk.
3. `helixc/frontend/typecheck.py` — multiple edits:
   - lines 736-757: cycle-5 C4-3 / HIGH symmetric TyVar/TySize/
     TyUnknown filter on aty side in `_check_call_basic`.
   - lines 1330-1450: cycle-5 C4-8 / LOW Logic-domain tie callback
     marker thread (`logic_domain_active[0]` mutable flag).
   - lines 2106-2209: cycle-5 C4-7 / F6 outer src/tgt threading in
     `_check_cast_compat` so trap-28604 preserves `&` ref prefix.
   - lines 2248-2312: cycle-8 C7-1 drop of G2 TyMemTier × (TyVar|
     TySize) carve-out + cycle-5 F2 / F3 / F4 doc-only notes
     for sub-domain matrix deferral.
   - lines 2379-2419: cycle-5 F7 / F8 / F9 / F10 new `_fmt_size`
     helper + tensor/tile/array shape rendering switch.
4. `helixc/check.py` lines 246-338 — cycle-5 C4-6 (MEDIUM)
   exception classifier + drain-failure isolation, modified at
   cycle 9 to drop the ImportError arm (closes C8-1) and route
   env-error printing through `_emit_env_error` (closes C8-2,
   strips a pre-existing `helixc:` prefix from a callee-formatted
   FileNotFoundError message).
5. `helixc/tests/test_typecheck.py` lines 1228-1572 — regression
   tests for cycle-5 C4-7 (ref-prefix preservation) + cycle-5
   C4-6 (FileNotFoundError + UnicodeDecodeError clean
   classification).
6. `helixc/tests/test_struct_mono.py` lines 805-857 — end-to-end
   test for cycle-5 C4-4 nested-turbofish substitution.
7. `helixc/tests/test_autodiff.py` lines 562-680 — cycle-5 C4-5
   tests for `Continue` / `TileLit` no-false-positive AD warning,
   plus the cycle-5/6 D2 polarity-revert assertion in
   `test_c6_revert_c4_2_literal_binary_no_false_trap`.
8. `helixc/tests/test_codegen.py` lines 3487-3508 — assertion
   inversion for D2-revert (i32-returning Call-RHS closure capture
   now succeeds with rc=3 instead of trap 132).

**Method**:

1. Loaded prior context from cycle-7 codereview doc, cycle-8
   codereview doc (persisted at 5d1ca24), and cycle-8
   silent-failures doc (which documented C8-1 + C8-2 against
   the cycle-8 classifier — closed at cycle 9 fix-sweep).
2. `git diff b8e047e..HEAD -- helixc/` covering 8 files
   (parser.hx, monomorphize.py, typecheck.py, check.py, plus 4
   test files).
3. For each touched file: read the changed region at HEAD plus
   ±20 lines of surrounding context. Verified the changes against
   the cumulative invariant set (post-cycles-1-through-7).
4. Walked the `_check_call_basic` symmetric filter: verified that
   pty=TyPrim('i32'), aty=TyVar('T') now defers via the
   `not isinstance(aty, (TyVar, TySize, TyUnknown))` clause,
   preventing the false-positive "expects i32, got T" during
   pre-mono body-typecheck of generic-adapter patterns.
5. Walked the `Monomorphizer.run` iteration: verified the
   two-loop pattern (non-generic items + promoted clones) reaches
   fixed-point on `fn id[T]; fn caller[U] -> U { id::<U>(v) };
   fn main { caller::<i32>(7) }` correctly producing id__i32 +
   caller__i32 and rewriting nested turbofish into mangled
   non-turbofish callees in two passes.
6. Walked `_check_cast_compat` outer-src/outer-tgt threading for
   `&Foo as &Bar` (1 peel) and `&&Foo as &&Bar` (2 peels):
   verified the outer types are preserved through recursion and
   the final error diagnostic prints `&Foo` / `&&Foo` rather
   than the peeled inner.
7. Walked the `_handle_binary` Logic-domain marker pattern:
   `logic_domain_active = [False]` is a per-call local
   (single-threaded; closure-captured reference; mutated only
   inside `_handle_binary`); the `_tie_cb` closure synchronously
   reads it during `_widen_diff_inner` and the flag is reset to
   False before exiting the Logic branch.
8. Walked `_fmt_size`: verified the three input classes
   (TyPrim with `size_N` prefix → return numeric suffix; TySize →
   return name; other → fall through to `_fmt`). Probed
   edge inputs: empty `size_` suffix → empty string (sub-threshold,
   AST never produces this); non-numeric `size_abc` → `'abc'`
   (sub-threshold, AST never produces this); TySize already
   contains `size:` prefix (no — TySize is constructed with raw
   names).
9. Ran the full `helixc/tests/test_typecheck.py` +
   `test_struct_mono.py` + `test_autodiff.py` suite at HEAD:
   183/183 PASS, including the four new regression tests:
   `test_c4_4_nested_turbofish_end_to_end_no_unresolved_generic_param`,
   `test_c4_7_cast_diagnostic_preserves_ref_prefix`,
   `test_c4_6_filenotfound_not_attributed_as_compiler_bug`,
   `test_c4_6_unicode_decode_error_clean_message`.

**Reporting threshold**: confidence ≥ 80 (strict criterion per
user directive 2026-05-10).

**Result**: **0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW) at
or above the confidence-80 reporting threshold.**

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| —     | —        | —          | —         | (none at or above threshold) |

**Cycle 8 Audit C (revised at HEAD 6968755): CLEAN — 0 findings at
the confidence-80 threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts
CLEAN only when zero findings of ANY severity at or above the
audit threshold. **This cycle qualifies as clean.**

---

## Cycle-7 + cycle-8 + cycle-9 finding closure verification

### C7-1 (LOW from cycle-7 silent-failures): G2 TyMemTier × (TyVar|TySize) carve-out leaks silent-acceptance to body/let/if/match-arm sites — **CLOSED**

Verified at HEAD typecheck.py lines 2273-2276: the carve-out arms
are gone. The both-TyMemTier same-tier arm (2273-2274) is unchanged;
the broad-or reject arm (2275-2276) is unchanged. Generic-call
defer is preserved via the call-boundary pre-filter in
`_check_call_basic` (lines 746-752).

### C4-1 / F1 (CRITICAL from cycle-5 silent-failures): cycle-3 D2 Call-RHS sentinel-12 produces SIGILL on i32-returning-fn capture pattern — **CLOSED**

Verified at HEAD parser.hx lines 2322-2349: the `val_tag == 16`
arm is gone; nothing remains between the val_tag == 41 (u16
literal) arm and the closing brace cascade. Both the cycle-4
broadening and the cycle-3 Call-only sentinel are gone.
test_codegen.py:3505 now asserts rc=3 (success) for the
`let pi = get_pi(); let c = |y| y + pi; c(0)` pattern when
`get_pi() -> i32`, replacing the prior rc=132 (SIGILL) assertion.

### C4-3 (HIGH from cycle-5 silent-failures): asymmetric TyVar filter in `_check_call_basic` produces false-positive on generic-adapter pattern — **CLOSED**

Verified at HEAD typecheck.py:746-747: the elif now filters BOTH
pty and aty for TyVar/TySize/TyUnknown. The canonical
`fn use_x[T](v: T) -> i32 { check_x(v) }` body-typecheck (where
the actual arg type is TyVar('T') and the expected param type is
TyPrim('i32')) now defers correctly to mono.

### C4-4 (HIGH from cycle-5 silent-failures): D9's nested-turbofish substitution was paper-only — **CLOSED**

Verified at HEAD monomorphize.py:462-487: the `Monomorphizer.run`
iteration uses a two-loop pattern. Generic-fn bodies are NOT
walked at top level (`if isinstance(item, A.FnDecl) and not
item.generics`). Clones are promoted into a separate walk set
after each pass so subsequent iterations cover their nested
turbofish. The end-to-end test
`test_c4_4_nested_turbofish_end_to_end_no_unresolved_generic_param`
verifies that `caller__i32`'s body calls `id__i32` (not the
unresolved `id__U`).

### C4-5 (HIGH from cycle-5 silent-failures): `_inline_lets` catch-all 85001 warn fires on Continue / TileLit — **CLOSED**

Two new regression tests in test_autodiff.py:565-606 assert
`_inline_lets(A.Continue(...), {})` and
`_inline_lets(A.TileLit(...), {})` do NOT emit 85001 warnings.
Both tests PASS at HEAD.

### C4-6 (MEDIUM from cycle-5 silent-failures): `except Exception` mis-attributes user-environment errors as compiler bugs — **CLOSED in cycle 8, refined in cycle 9**

Verified at HEAD check.py:299-318: the typed-except cascade has
two env-error arms (FileNotFoundError + siblings;
UnicodeDecodeError) plus the broad Exception arm. ImportError is
no longer in the cascade — it falls through to the
broad Exception arm with the "compiler bug" tagline + rc=1
(cycle-9 close of C8-1). Two regression tests in
test_typecheck.py:1530-1572 (
`test_c4_6_filenotfound_not_attributed_as_compiler_bug`,
`test_c4_6_unicode_decode_error_clean_message`) PASS at HEAD.

### C4-7 / F6 (MEDIUM from cycle-5 silent-failures): `&Foo as &Bar` trap-28604 diagnostic prints peeled inner instead of outer with ref prefix — **CLOSED**

Verified at HEAD typecheck.py:2106-2209: `_check_cast_compat`
takes new `_outer_src` / `_outer_tgt` parameters threaded through
the post-peel recursive call. The final error message at line
2202-2204 uses `_outer_src` / `_outer_tgt`, preserving the `&`
prefix. The regression test
`test_c4_7_cast_diagnostic_preserves_ref_prefix` PASSES at HEAD.

### C4-8 (LOW from cycle-5 silent-failures): `_widen_diff_inner` tie callback drops `[Logic-domain]` suffix — **CLOSED**

Verified at HEAD typecheck.py:1349, 1430-1435: a per-call mutable
flag `logic_domain_active = [False]` is set True before
`_widen_diff_inner` in the Logic branch and reset False after.
The `_tie_cb` reads the flag and appends ` [Logic-domain]` to
the same-rank-tie warn when the flag is set.

### C8-1 (MEDIUM from cycle-8 silent-failures): cycle-5 ImportError arm catches internal-pipeline ImportError as user-env-error rc=2 — **CLOSED in cycle 9**

Verified at HEAD check.py:299-318: ImportError is no longer an
explicit arm; it falls through to the broad `except Exception`
arm with the correct rc=1 + "compiler bug" tagline.

### C8-2 (LOW from cycle-8 silent-failures): pre-formatted `helixc:` FileNotFoundError prefix double-prints — **CLOSED in cycle 9**

Verified at HEAD check.py:246-255: the new `_emit_env_error`
helper strips a leading `helixc:` prefix before re-prefixing.
Used by both env-error arms (lines 301 + 304).

### F2 / F3 / F4 (cycle-5 MEDIUM, doc-only): `_compatible` sub-domain matrix limitations — **DOCUMENTED**

Verified at HEAD typecheck.py:2266-2272, 2290-2295, 2300-2308:
three new comment blocks document Phase-0 limitations (TyMemTier
strict-equality on tier, TyDiff single-domain, TyLogic provenance
not in structural matcher). No code change — pure deferred
enhancement docs.

### F7 / F8 / F9 / F10 (cycle-5 LOW, helper + format polish): `_fmt_size` helper + shape-element rendering — **CLOSED**

Verified at HEAD typecheck.py:2379-2391 (`_fmt_size`),
2401-2419 (uses in TyTensor / TyTile / TyArray `_fmt` arms). The
helper renders `size_3` → `3`, `TySize('N')` → `N`, fall-through
to `_fmt` for non-size types. Concrete Phase-0 conventions
guarantee size_N is always numeric so the edge inputs noted
below-threshold (non-numeric suffix, empty suffix) are
unreachable.

---

## Specific HEAD changes audited (cycle 8 + cycle 5 cluster + cycle 9 deltas)

1. **parser.hx final D2 revert** — both reverted gates gone;
   parser-side closure-capture inference reverted to literal-RHS-
   only (cycle-2 baseline state). **PASS.**

2. **monomorphize.py `Monomorphizer.run` two-loop iteration** —
   the non-generic-only top-level walk avoids the paper-only D9
   regression. Promoted-clone walk set + `if fn not in promoted`
   filter terminate at fixed point. The `if self.instantiated`
   block at the end of the while-loop promotes any newly-added
   clones (O(n^2) on promoted-list lookup, but n is the
   instantiation count — bounded by the source's distinct
   turbofish set, typically < 10). **PASS.**

3. **typecheck.py `_check_call_basic` symmetric filter at line
   746-747** — both pty and aty filtered for TyVar/TySize/
   TyUnknown. The TyPrim-vs-TyPrim primary arm at line 710-720
   handles the concrete-vs-concrete case; the elif handles the
   non-prim and mixed-kind cases where neither side is generic.
   Defer path covers pty=TyPrim, aty=TyVar (and vice versa).
   **PASS.**

4. **typecheck.py `_widen_diff_inner` Logic-domain flag pattern
   at line 1349, 1430-1435** — per-call mutable list (Python
   closure-over-mutable-state idiom). Single-threaded
   execution; no race. The flag is set True only inside the
   Logic branch and reset to False on exit. The D branch (line
   1385-1388) does NOT set the flag, so the suffix correctly
   stays empty when the tie fires inside D-domain. **PASS.**

5. **typecheck.py `_check_cast_compat` outer threading at lines
   2106-2209** — `_outer_src` / `_outer_tgt` default to None;
   first-call init at lines 2137-2140 captures the user-visible
   form (with `&` prefix preserved). Recursive call after peel
   passes the outer types through. The error message at line
   2202-2204 uses the outer types. Trap-28803 depth-guard
   message (line 2166-2173, 2178-2186) still uses the
   un-threaded message because it's an internal depth-cap
   diagnostic, not a user-visible cast-matrix-pair issue. **PASS.**

6. **typecheck.py cycle-8 C7-1 drop of G2 carve-out at lines
   2273-2276** — verified per the prior cycle-8 codereview doc.
   Both arms gone. Cross-walked every `_compatible` call site:
   defer path preserved at call boundary (`_check_call_basic`)
   via TyVar pre-filter; hard-error path now fires at body /
   let / if / match-arm sites for TyMemTier × TyVar pairs,
   matching C7-1 rationale. **PASS.**

7. **typecheck.py `_fmt_size` helper at lines 2379-2391** —
   handles `TyPrim('size_N')` (returns suffix), `TySize` (returns
   name), and falls through to `_fmt` for other types. Used in
   TyTensor / TyTile / TyArray `_fmt` arms. The wrappers correctly
   call `_fmt_size` only on size-position elements, not on dtype
   / elem / inner. **PASS.**

8. **check.py `_emit_env_error` helper at lines 246-255** —
   strips a single leading `helixc:` prefix before reprefixing.
   Idempotent: re-applying to an already-clean message is a
   no-op for the strip (no `helixc:` prefix to strip). Edge
   case: a message starting with `"helixc: helixc: ..."` after
   one strip still has `"helixc: ..."`; the print reprefixes to
   `"helixc: helixc: ..."` — same double-prefix result. Below
   threshold (callees don't double-prefix in practice). **PASS.**

9. **check.py cycle-9 exception cascade at lines 299-318** —
   ImportError no longer has an explicit arm. FileNotFoundError
   + siblings + UnicodeDecodeError route to `_emit_env_error`
   with rc=2; everything else routes to the broad Exception arm
   with the "compiler bug" tagline + rc=1. Exit-code contract
   preserved per the file-header docstring. **PASS.**

10. **Drain-failure isolation at lines 319-337** — the finally
    block wraps `_drain_ad_warnings` and `_drain_ad_init` in a
    try/except. A drain crash now prints a "warning: AD-warning
    drain failed:" stub and preserves the primary rc. Below
    threshold: a successful rc=0 with `-Wad=error` policy that
    the drain would have elevated to rc=1 will now stay at rc=0
    on drain crash (sub-threshold — drain crash is itself a
    compiler-internal bug requiring investigation). **PASS.**

---

## What was checked and found below threshold

- **`Monomorphizer.run` O(n^2) `fn not in promoted` lookup**:
  for each pass, iterates `self.instantiated.items()` and tests
  membership in `promoted` (a list). For n distinct
  instantiations, O(n^2) total. Real-world n < 10, no
  performance issue. Could be a set, but list preserves order
  which `_rewrite_calls_in_block` iteration relies on for
  deterministic walk order. **Confidence 25**, below threshold.

- **`Monomorphizer.run` clones never removed from `instantiated`**:
  the dict only grows. If a clone's body is rewritten (so the
  clone calls a different mangled name), the old clone reference
  stays. Memory-only concern; no correctness regression because
  rewrite uses `is not` identity check. **Confidence 30**, below
  threshold.

- **`_emit_env_error` strips only one `helixc:` prefix**: a
  pathological `"helixc: helixc: helixc: ..."` message still
  produces a `"helixc: helixc: ..."` print. Callees in
  practice never produce more than one `helixc:` prefix
  (verified by walking parser.py:1587 and check.py raise sites).
  **Confidence 30**, below threshold.

- **`_emit_env_error` would strip a legitimate user-message
  prefix**: if a user-content string in a filename or path
  literally begins with `helixc:` (e.g., a path
  `helixc:something/file.hx`), the leading would be stripped.
  Pathological / unrealistic input. **Confidence 25**, below
  threshold.

- **`_check_cast_compat` outer threading doesn't apply to trap-
  28803 (depth-guard) diagnostic**: the recursion-depth and
  ref-peel-depth diagnostics use the pre-formatted message
  without the user-visible outer types. Defensible —
  trap-28803 is a compiler-internal cap, not a per-cast matrix
  rejection, and shouldn't expose a confusing "&Foo as
  &Bar" pair when the real issue is depth. **Confidence 40**,
  below threshold.

- **`_check_cast_compat` `_outer_src is None` initial-call
  detection conflates first-call with explicit None pass**:
  callers cannot intentionally pass `_outer_src=None` to mean
  "no outer" (they'd have to pass a sentinel). Real callers
  don't do this — `_check_cast_compat` is invoked from one
  site (Cast expr handler) with no explicit outer args.
  **Confidence 25**, below threshold.

- **`_fmt_size` for `TyPrim('size_')` (empty suffix)** would
  return `''`. The AST construction at monomorphize.py:257
  always produces non-empty suffix via `f"size_{repl.value}"`
  where `repl.value` is an int. Edge case unreachable.
  **Confidence 20**, below threshold.

- **`_fmt_size` for `TyPrim('size_abc')` (non-numeric suffix)**
  would return `'abc'`. Not produced by Phase-0 AST. Edge case
  unreachable. **Confidence 20**, below threshold.

- **`logic_domain_active = [False]` mutable closure pattern**:
  Pythonic idiom for outer-scope mutation. No threading concern
  in typecheck (single-threaded). The flag is correctly scoped
  to the `_handle_binary` call (per-call local). Below threshold
  in practice. **Confidence 25**, below threshold.

- **`_handle_binary` two-arm pattern (D branch + Logic branch)
  shares `tie_fired`, `logic_domain_active`, `_tie_cb` —
  intentional cross-arm coordination**: the D branch (lines
  1375-1407) and the Logic branch (lines 1408-1443) are
  mutually exclusive via the `if (l_is_diff or r_is_diff)`
  vs. `elif (l_is_logic or r_is_logic)` chain. Only one branch
  fires per binop. Cross-branch state-sharing is intentional —
  if a future branch were added that ALSO wanted Logic-domain
  semantics, it would need to set the flag. **Confidence 35**,
  below threshold.

- **No regression test for the cycle-8 C7-1 drop**: a
  `_compatible(TyMemTier(W, i32), TyVar('T'))` returns-False
  assertion would document the post-cycle-8 contract. The
  cycle-5/7 test additions and the cycle-7 G2 close also had
  no such test. Test-density precedent established in cycles
  6-8 for structural-matcher arm tweaks. **Confidence 55**,
  below threshold.

- **No test for the `_emit_env_error` prefix-strip**: the
  cycle-9 close of C8-2 added the helper but no targeted test
  for the strip behavior. The cycle-9 commit note says 264
  tests pass at HEAD, and the C4-6 regression tests
  (test_typecheck.py:1530-1572) verify the env-error rc=2
  classification works — but not specifically the
  double-prefix avoidance. **Confidence 60**, below threshold.

- **`Monomorphizer.run` doesn't drop the original generic
  FnDecls**: the comment at line 488-489 says "keep original
  generic fns intact so legacy un-turbofished call sites
  keep resolving (cycle-3 backward-compat)". This means
  generic-fn bodies appear in `prog.items` but are not
  walked — they're "stale" relative to the rewritten main /
  caller bodies. Downstream lowering must skip them (per
  the cycle-3 backward-compat contract). Not a regression
  here; documented existing behavior. **Confidence 30**,
  below threshold.

- **Trap ID consistency**: no new traps added in this delta.
  trap-28604 (legacy invalid-cast) and trap-28803
  (cast-matrix-recursion-depth) are unchanged. **Confidence
  20**, below threshold.

- **Dead imports / dead code**: walked the touched regions in
  typecheck.py / monomorphize.py / check.py / parser.hx for
  leaked imports or now-unused helpers. None found. The
  `_emit_env_error` helper is used in two arms (lines 301,
  304). The `_fmt_size` helper is used in three TyFmt arms
  (lines 2406, 2410, 2419). The `_outer_src`/`_outer_tgt`
  params are used in the final error message. The
  `logic_domain_active[0]` flag is read by `_tie_cb` and
  written by the Logic branch. All clean. **Confidence 20**,
  below threshold.

- **FLAT prefix-trap convention (AST_TAG * 1000 + sub_id)**:
  the existing traps in this delta (28604, 28803, 76003,
  85001, 24200) follow legacy conventions, not the FLAT
  prefix-trap scheme. No new traps added in this delta, so
  no convention violation. **Confidence 15**, below threshold.

---

## Open prior findings (not re-flagged this cycle)

Per the cumulative cycles-1-through-8 silent-failures /
type-design / code-review tracking:

- **audit-C4-2** (HIGH — D9 paper-only): **CLOSED in cycle 5**
  (HIGH cluster commit c30b8f5) per the cycle-5 C4-4 fix at
  monomorphize.py:462-487.
- **audit-C4-7 / C5-7** (MEDIUM — check.py `except Exception`):
  **CLOSED in cycle 8** (5d1ca24), refined in cycle 9 (6968755)
  to drop the ImportError arm + add the prefix-strip helper.
- **audit-C7-1** (LOW — G2 carve-out top-level placement):
  **CLOSED in cycle 8** (5d1ca24) per the carve-out drop at
  typecheck.py:2273-2276.
- **audit-C8-1** (MEDIUM — ImportError mis-classified as env
  error): **CLOSED in cycle 9** (6968755) per the explicit-arm
  drop at check.py:299.
- **audit-C8-2** (LOW — double `helixc:` prefix on callee-
  formatted FileNotFoundError): **CLOSED in cycle 9** (6968755)
  per the `_emit_env_error` helper.
- **audit-C4-1 / D2 critical regression** (CRITICAL — Call-RHS
  SIGILL on i32-returning fn): **CLOSED in cycle 5** (commit
  75e2209) per the full revert of cycle-3 D2 at parser.hx:2322.
- **audit-C4-3** (HIGH — asymmetric TyVar filter in
  `_check_call_basic`): **CLOSED in cycle 5** (HIGH cluster
  commit c30b8f5) per the symmetric filter at
  typecheck.py:746-747.
- **audit-C4-4** (HIGH — Monomorphizer iteration order):
  **CLOSED in cycle 5** (commit c30b8f5) per the two-loop
  iteration at monomorphize.py:462-487.
- **audit-C4-5** (HIGH — `_inline_lets` Continue/TileLit false
  positive AD warn): **CLOSED in cycle 5** (HIGH cluster commit
  c30b8f5) per the regression tests at
  test_autodiff.py:565-606.
- **audit-C4-6** (MEDIUM — check.py classifier): **CLOSED**
  per above.
- **audit-C4-7** (MEDIUM — cast ref-prefix preservation):
  **CLOSED in cycle 5** (MEDIUM cluster commit ae1ebff) per
  the outer-src/outer-tgt threading at typecheck.py:2106-2209.
- **audit-C4-8** (LOW — Logic-domain tie suffix): **CLOSED in
  cycle 5** (LOW cluster commit 164a530) per the
  logic_domain_active flag at typecheck.py:1349, 1430-1435.

Still-open carryovers from prior cycles (NOT in this delta's
scope, NOT re-flagged this cycle):

- **audit-C4-4-deferred** (paper-only D9): superseded by
  cycle-5 C4-4 close.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open; not addressed in this delta.
- **f-string / `_fmt` test-density gaps**: documented in
  prior cycles, not blocking.

No code-review regressions introduced by the cycle-5 cluster
fixes, the cycle-8 G2 drop, or the cycle-9 refinement:

- `_compatible` recursion termination still holds.
- `_size_compatible` (cycle-7 helper) still works.
- `_check_call_basic` still emits one (not two) diagnostics on
  cross-kind violations.
- `check.py` exit-code contract preserved (0/1/2 mapping
  unchanged across cycle-5 add, cycle-9 refinement).
- `_drain_ad_warnings` contract preserved (rc-elevate-on-error
  path now wrapped, but the rc=1 elevation still fires when
  the drain succeeds and policy is `error`).
- `_handle_binary` widen-then-warn contract preserved; the
  cycle-5 C4-8 marker thread only refines the suffix, not the
  emit gate.
- `_check_cast_compat` trap-28604 behavior preserved; the
  cycle-5 C4-7 thread only refines the diagnostic rendering.
- `Monomorphizer.run` fixed-point termination preserved; the
  cycle-5 C4-4 two-loop pattern still terminates when
  `self.instantiated` stops growing.

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16
baselines unchanged from cycle-1 status; cycle-1 through cycle-8
findings all marked CLOSED by their respective fix-sweep commits
(except for the still-open MEDIUM/LOW carryovers documented above
that are NOT in this audit's scope).

---

## Cycle 8 status

**Cycle 8 Audit C (revised at HEAD 6968755): CLEAN — 0 findings
(0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW) at or above the
confidence-80 reporting threshold.**

Strict-zero rule per user directive 2026-05-10. The cycle-8 fix-
sweep (G2 drop) + the cycle-5 cluster fix-sweep (C4-1 / C4-3 /
C4-4 / C4-5 / C4-6 / C4-7 / C4-8 / F2..F10) + the cycle-9 fix-
sweep (C8-1 / C8-2) together produced a well-targeted, well-
scoped set of closures. The review found no regressions
introduced by any of the deltas.

The below-threshold notes (no negative test for cycle-8 C7-1
drop, no test for `_emit_env_error` strip behavior, O(n^2)
`fn not in promoted` membership test, drain-failure rc
suppression on success, etc.) are documented for future cycles
but do not block this cycle's clean status. The prior cycle-8
codereview doc (at 5d1ca24, persisted in commit 68bdb7f) is
preserved as-is; this revised doc widens the scope to the full
b8e047e..HEAD delta.

Cycle counter advances provided the cycle-8 silent-failures
(2 findings — CLOSED by cycle-9 fix-sweep at HEAD) and cycle-8
type-design audits are re-evaluated at HEAD with the same
strict-zero criterion.
