# Stage 28.8 Cycle 6 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: c3f26ef (read-only audit). Cycle-5 fix-sweep range
960303b..c3f26ef (1 squashed fix-sweep commit covering revert of
C4-2 + closure of C5-1..C5-4 + F1..F6).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically re-audits the cycle-6
fix-sweep changes for fresh silent windows introduced by the
fixes themselves.
**Trigger**: pre-Stage-29 audit gate — Cycle 6 of 5+ (the gate
re-arms each time a cycle is not clean). Re-audits same scope
after cycle-6 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Per the user directive for cycle 6,
findings already documented in cycles 1-5 are NOT re-flagged unless
they CHANGED in the cycle-6 fix-sweep.

**Method**:
1. Read prior cycle silent-failure docs (cycle 1 — 13 findings;
   cycle 2 — 6 findings; cycle 3 — 6 findings; cycle 4 — 8
   findings; cycle 5 — 4 findings) to avoid re-flagging
   already-documented findings.
2. Walked `git show c3f26ef` — the single cycle-6 fix-sweep
   commit. Read the diff for each of: parser.hx (REVERT C4-2),
   typecheck.py (F1 cascade arm + F2 D-vs-bare + TRAP_* f-strings),
   autodiff.py (C5-3 TileLit recursive walk), monomorphize.py
   (TRAP_* f-strings), backend/x86_64.py (C5-4 abort).
3. For each cycle-6 fix's diff, traced data flow forward to check
   whether the fix opened a fresh silent window, left a fix
   incomplete (paper-only), compounded a prior-cycle regression,
   or over-corrected.
4. Spot-checked the new code for: dispatch holes, state-leak
   after exception, error-channel reach (warn-vs-abort), and
   false-acceptance broadening at every other `_compatible`
   callsite (since the new top-level TyVar/TySize defer arm is
   the broadest change in cycle 6).
5. Cross-checked the cycle-6 fix coverage against the still-open
   carryover findings from cycles 1-5 to identify which carryovers
   were actually CLOSED by cycle 6 vs which remain open.

**Result**: **1 new finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW)**
— Cycle 6 NOT clean. The fix-sweep makes EXCELLENT progress —
all four cycle-5 findings (C5-1 revert; C5-2 cascade arm;
C5-3 TileLit walk; C5-4 abort) are CORRECTLY closed at the
mechanism level. C5-1 in particular is a clean revert that
restores the dominant idiom (`let a = 10 + 5; let c = |x| x + a;
c(5)`) to working order. C5-2 closes via the top-level TyVar/TySize
defer arm, which simultaneously closes the cycle-4 carryover
audit-C4-3 (TyVar-arg-to-TyPrim-param silent-pass — was already
guarded at the call boundary, but the inner-recursion path now
also defers correctly).

The one new finding (C6-1) is the SECONDARY EFFECT of the C5-2
top-level `TyVar`/`TySize` defer arm: it broadens beyond the
cycle-5 target (`_compatible(TyTensor.shape[i], TyTensor.shape[i])`
recursion) to EVERY `_compatible` callsite where one side is
TyVar/TySize, including `_compatible(body_ty, sig.ret)` (line
1142), `let y: T = expr` (line 1164), `if/else branches differ`
(line 1550), and `match arm` (line 1576). At those sites, a
TyVar appearing in the declared type now silently accepts any
value type — a fresh silent-acceptance broadening adjacent to
the intended fix.

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

(none)

---

## MEDIUM FINDINGS

### Finding C6-1: cycle-6 C5-2 / F1 top-level `TyVar`/`TySize` defer arm in `_compatible` broadens too far — at the function-body / let-declared / if-else / match-arm `_compatible` callsites, a declared type containing a free TyVar now silently accepts any value type instead of emitting "body type T does not match return type i32"

**Location**:
- helixc/frontend/typecheck.py:2166-2179 (the new top-level TyVar/TySize defer arm)
- helixc/frontend/typecheck.py:1142 (`_check_fn_decl`: `_compatible(body_ty, sig.ret)`)
- helixc/frontend/typecheck.py:1164 (`_check_stmt`/Let: `_compatible(value_ty, declared)`)
- helixc/frontend/typecheck.py:1550 (`_check_expr`/If: `_compatible(t, e)` for if/else branches)
- helixc/frontend/typecheck.py:1576 (`_check_expr`/Match: `_compatible(first, t)` for match arms)
**Severity**: MEDIUM
**Category**: cycle-6-fix-introduced silent-acceptance broadening (over-correction)
**Stage**: 28.8 cycle-6 commit c3f26ef (C5-2 / F1)

**Description**:
The cycle-6 fix-sweep added a top-level cascade-safe defer arm to
`_compatible`:

```python
def _compatible(self, a: Type, b: Type) -> bool:
    if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
        return True
    # Audit 28.8 cycle 6 C5-2 / F1: cascade-safe arm for TyVar /
    # TySize. ... Defer to mono substitution for these — same
    # cascade-safe rule as TyUnknown.
    if isinstance(a, (TyVar, TySize)) or isinstance(b, (TyVar, TySize)):
        return True
    ...
```

The fix's CORRECT target is the inner-shape-recursion path:
`_compatible(TyTensor(shape=[TySize('N')]), TyTensor(shape=[TyPrim('size_3')]))`
recurses through the cycle-4 E1 / cycle-4 C4-4 structural arms
into `_compatible(TySize('N'), TyPrim('size_3'))`, which pre-fix
fell through to `a == b` → False, mis-firing the generic-tensor
call boundary. With the new top arm that recursive compare returns
True correctly.

But the fix is implemented at the TOP of `_compatible`, so it
ALSO defers at every OTHER callsite of `_compatible`. Pre-cycle-6,
`_compatible(body_ty, sig.ret)` for `fn g[T]() -> T { 42 }` had:
- body_ty = TyPrim('i32') (from `42`)
- sig.ret = TyVar('T')
- Result: pre-cycle-6 fell through all the structural arms (no
  matching wrapper / composite combo) to `return a == b` → False
  → emitted "function 'g': body type i32 does not match return
  type T".

Post-cycle-6: `isinstance(b, TyVar)` → returns True immediately
→ silently accepts. The user gets no diagnostic that their
generic return-type signature is being satisfied by a concrete
i32 literal (which is almost always a type-pun bug — a polymorphic
function returning a fixed scalar can't be polymorphic).

Verified end-to-end via direct Python probe:

```python
>>> from helixc.frontend.typecheck import TypeChecker, TyVar, TyPrim
>>> from helixc.frontend import ast_nodes as A
>>> tc = TypeChecker(A.Program(module=None, items=[]))
>>> tc._compatible(TyPrim('i32'), TyVar('T'))
True   # post-cycle-6 (silently accepts)
       # pre-cycle-6 was False
```

The same broadening affects:

1. **Line 1142 — function body vs sig return**:
   `fn g[T](x: i32) -> T { x }` silently typechecks clean. Pre-fix:
   "function 'g': body type i32 does not match return type T".

2. **Line 1164 — let with declared type**:
   `fn h[T]() -> T { let y: T = 42; y }` silently accepts the
   `let y: T = 42` assignment. Pre-fix: "let 'y': declared T
   but value is i32".

3. **Line 1550 — if/else branches**:
   `fn k[T](b: bool, x: T) -> T { if b { x } else { 0 } }`
   silently accepts `if x else 0` where x is T and 0 is i32.
   Pre-fix: "if/else branches differ: T vs i32".

4. **Line 1576 — match arm bodies**:
   `fn m[T](x: T) -> T { match 1 { 1 => x, _ => 0 } }` silently
   accepts mismatched arm types. Pre-fix: "match arm 1 body type
   i32 incompatible with arm 0 type T".

5. **Line 1649 — Pattern bindings against scrutinee**: same
   silent acceptance for `let Some(v: T) = some_i32_value;`.

The cycle-5 audit's C5-2 recommendation suggested EITHER (a)
the top-level TyVar/TySize defer arm OR (b) per-shape-element
defer logic inside each shape-bearing structural arm. The
fix-sweep took option (a), which is broader and simpler but
opens these other silent windows. Option (b) would have been
strictly scoped to the shape-recursion path.

The call boundary check at line 736 already guards against
`pty: TyVar/TySize/TyUnknown` skipping `_compatible` entirely
(this guard predates cycle 6 — it's the cycle-3 D1 elif arm),
so call sites are still correctly checked. But the other five
callsite categories are now silently broadened.

**Hidden errors**:
- Every `fn f[T]() -> T { 42 }` style polymorphic-return-from-
  monomorphic-literal silently typechecks where pre-cycle-6 it
  errored. Users get no signal that their generic-T return is
  being trivially satisfied by a non-generic body.
- Every generic `let y: T = const_literal` silently accepts.
  If T eventually resolves to a different type at the mono site,
  the user sees a confusing downstream error away from the let.
- `if b { x: T } else { 0: i32 }` silently picks the first arm's
  type as the if-result; pre-fix errored cleanly. Same for match.
- The cycle-6 fix-sweep does NOT add tests covering these adjacent
  silent windows. The new test `test_c5_2_compatible_tysize_cascade`
  asserts the SHAPE-recursion success (correct) but not the
  ADJACENT body/let/if/match callsite behavior.
- A future contributor reading the new top-level arm's comment
  ("Defer to mono substitution for these — same cascade-safe
  rule as TyUnknown") may not realize TyUnknown was guarded by
  the call boundary `_compatible` skip; TyVar/TySize have looser
  guarding at the body/let/if/match sites.

**Recommendation**:
1. Narrow the cycle-6 F1 fix to the shape-recursion path only,
   per the cycle-5-audit's option (b). Move the TyVar/TySize
   defer INSIDE each shape-bearing arm where shape elements are
   compared. Specifically inside the `TyArray`, `TyTensor`,
   `TyTile` structural arms:
   ```python
   if isinstance(a, TyArray) and isinstance(b, TyArray):
       if not self._compatible(a.elem, b.elem):
           return False
       sa, sb = a.size, b.size
       if (isinstance(sa, (TyVar, TySize, TyUnknown))
               or isinstance(sb, (TyVar, TySize, TyUnknown))):
           return True   # defer to mono
       return sa == sb or self._compatible(sa, sb)
   ```
   Same pattern for `TyTensor.shape` and `TyTile.shape` zips.
2. Remove the top-level TyVar/TySize defer arm so the body /
   let / if-else / match-arm `_compatible` callsites preserve
   their pre-cycle-6 diagnostic behavior.
3. Alternatively (if the top-level defer is preferred for
   simplicity): add explicit pre-checks at lines 1142, 1164,
   1550, 1576, 1649 that emit the diagnostic when EITHER side
   is `TyVar` / `TySize` AND the OTHER side is a concrete type
   that wouldn't unify under mono. The pre-checks belong at
   each callsite (semantic interpretation of TyVar at "body
   should match sig.ret" is "body must be polymorphic over T",
   not "body can be any T-compatible type").
4. Add regression tests covering:
   - `fn g[T]() -> T { 42 }` must error.
   - `fn h[T](x: T) -> T { let y: T = 42; y }` must error on the
     `let y: T = 42` assignment.
   - `fn k[T](b: bool, x: T) -> T { if b { x } else { 0 } }`
     must error on the if/else mismatch.

**Trap-id**: n/a (typecheck silent-accept, no trap-id).

---

## LOW FINDINGS

(none)

---

## Cycle 6 fix-sweep re-verification

Each cycle-6 fix-sweep change was inspected for paper-only fixes,
silent windows, false positives, and state-leak. The cycle-6
fix-sweep landed as a single commit (c3f26ef) covering: REVERT of
cycle-4 C4-2 broadening + F1..F6 (closes C5-1..C5-4 plus F2 / F5
/ F6 housekeeping).

| fix-sweep label | What changed | Audit-doc cross-ref | C6 verdict |
|---|---|---|---|
| REVERT C4-2 | parser.hx tag-12 sentinel back to Call-only (val_tag == 16) | C5-1 | **closed** — the per-RHS-class broadening is gone; the dominant idiom `let a = 10 + 5; let c = |x| x + a` no longer SIGILLs |
| F1 / C5-2 | `_compatible(TyVar | TySize, _) -> True` top arm | C5-2 | mechanism correct at shape-recursion target; **but opens C6-1** adjacent silent windows |
| F2 | D<T> + T same-inner asymmetric wrap warns | (cycle-5 type-design F2) | OK — symmetric with cycle-4 E2 Logic case; emits AD002 warning on previously silent path |
| F3 / C5-4 | x86_64.py monomorphize_safe diags → `sys.exit(1)` | C5-4 | **closed** — half-mutated prog state no longer reaches grad_pass/typecheck/codegen on the binary-emit path; the docstring at monomorphize_safe still says "callers that don't care can ignore diags; the count is 0" which is now stale (the only caller doesn't ignore) but no silent failure remains |
| F4 / C5-3 | `_inline_lets` TileLit recursive walk (shape + memspace) | C5-3 | **closed** — `_inline_lets(TileLit(shape=[Name('N')]), {'N': IntLit(8)})` now returns a TileLit with `shape=[IntLit(8)]` |
| F5 / F6 | (housekeeping in commit message — covered by REVERT C4-2) | C5-1 | already covered above |
| audit-C C5-1 | TRAP_* f-string interpolation (28801/28802/28803) | (housekeeping) | OK — constants imported at module scope; f-string in-scope; trap-ids.md "every TRAP_* must have at least one reader" invariant satisfied |
| audit-C C5-2 | Regression tests added (test_c4_1_path / test_c4_3_inline_lets_if_cond / test_c5_2_compatible_tysize_cascade / test_c6_revert_c4_2) | (housekeeping) | OK — tests assert intended post-fix behavior |

### Specific re-verifications from the audit instructions

- **REVERT of cycle-4 C4-2 (parser.hx)**: verified the broadening
  block is gone via direct file inspection. Lines 2330-2374 in
  the diff show only the comment block remains + the single
  `inferred_ty_tag = 12;` for `val_tag == 16`. The if/else cascade
  for val_tag 1/6/19/20/21/22/23 is removed. test_codegen.py:3498
  still asserts `c(0) == 132` for the Call-RHS case (the still-
  open cycle-4-audit C4-1 critical is documented as a carryover,
  not re-flagged in cycle 6 per the user directive).
- **F1 (`_compatible` TyVar/TySize top arm)**: shape-recursion
  path works correctly — `_compatible(TyArray(elem=i32,
  size=TySize('N')), TyArray(elem=i32, size=TyPrim('size_3')))`
  → True. But the arm sits at the TOP of `_compatible`, so it
  also defers at body / let / if-else / match-arm sites. See
  C6-1.
- **F2 (D-vs-bare same-inner)**: gate broadened from
  `inner_mismatch` alone to `inner_mismatch OR (l_is_diff !=
  r_is_diff)`. Probed via Python repro — `D<f64> + f64` now
  hits `_ad_warn_mixed_inner` with "(one side D-wrapped, other
  bare)" extra. Pre-fix the same input silently produced
  `D<f64>` with no diagnostic. Surface reach: gated by D-binop
  parser support; today user-reachable via explicit autodiff
  call. No regression introduced. Symmetric with cycle-4 E2
  Logic case (which was checked clean in cycle 5).
- **F3 (x86_64.py mono_diags abort)**: the call site now
  emits `error: fn-mono: ...` (not `warning:`) and calls
  `sys.exit(1)` before reaching grad_pass / typecheck / codegen.
  Half-mutated `prog` state never propagates downstream on the
  binary-emit path. Verified via the diff at x86_64.py:3032-
  3036. NOTE: `check.py` (the surface tool) does NOT call
  `monomorphize_safe` at all (only `monomorphize_structs`),
  so a Helix program with a fn-mono `ShapeFoldError` trigger
  via `helixc check foo.hx` still produces no diagnostic at
  all — but this is a still-open cycle-5 carryover (deferred
  observation), not a new cycle-6 finding.
- **F4 (TileLit recursive walk)**: Probed via Python repro —
  `_inline_lets(TileLit(shape=[Name('N')], memspace=Name('REG'),
  dtype=TyPrim('f32'), init='zeros'), {'N': IntLit(8)})` returns
  `TileLit(shape=[IntLit(8)], memspace=Name('REG'), ...)`.
  Let-bindings now substitute correctly. The fix preserves the
  `dtype` (a TyNode, not an Expr) and `init` (a str literal)
  unchanged — correct (those have no Expr children).
- **TRAP_* f-string interpolation**: probed at all five call
  sites (typecheck.py:591/596/606/611 for TRAP_ARRAY_SIZE_NEG;
  typecheck.py:2109/2122 for TRAP_CAST_MATRIX_RECURSION_DEPTH;
  monomorphize.py:117/125 for TRAP_SHAPE_FOLD_ZERO_DIV). All
  resolve correctly to module-level constants 28802 / 28803 /
  28801. No silent failures — the f-string renders the same
  integer the literal previously did, but now the constant is
  the source of truth (grep-discoverable consumer).

### Carryover findings status (cycles 1-5)

The cycle-6 fix-sweep did NOT re-attempt the following
still-open carryover findings (acknowledged in the cycle-5
audit's "Estimated remaining open findings going into cycle 6"
section):

| Carryover | Severity | Cycle-6 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — cycle-6 explicitly defers re-fix until parse-time constant folding or a typecheck pass before closure capture exists |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-7 (check.py `except Exception`) | MEDIUM | **still open** — not addressed |
| audit-C4-8 deferred (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |

These are NOT re-flagged as new cycle-6 findings per the user
directive (already documented in cycles 1-5, did not CHANGE in
cycle 6's fix-sweep). They remain in the open-findings ledger.

### Specific items checked clean in cycle 6 (no new finding)

- The cycle-4 C4-2 broadening is genuinely gone from parser.hx —
  REVERT is clean. The dominant idiom `let a = 10 + 5; let c =
  |x| x + a; c(5)` returns 20 (verified by the cycle-5 prior
  state diff; the broadening block in lines 2334-2374 was
  removed).
- TileLit walking is structurally correct — shape and memspace
  are walked; dtype (TyNode) and init (str) are correctly
  preserved as-is (no Expr children to walk).
- The `monomorphize_safe` wrapper still catches ONLY
  `ShapeFoldError` (specific, not bare Exception). The caller
  now aborts. The pre-fix C5-4 silent-miscompile window is closed.
- F2 (D-vs-bare) emits a warning where pre-fix it was silent —
  this is a louder-feedback fix, not a new silent window.
- TRAP_* f-string interpolation is wired correctly at all five
  call sites. No literal-vs-constant drift remains; the trap-ids.md
  audit-time invariant ("every TRAP_* must have at least one
  reader") is satisfied without changing the emitted diagnostic
  string (the f-string renders the same integer).
- The regression tests added by the fix-sweep
  (test_c4_1_path_no_false_positive, test_c4_3_inline_lets_if_cond_substituted,
  test_c5_2_compatible_tysize_cascade, test_c6_revert_c4_2_literal_binary_no_false_trap)
  do NOT codify any broken behavior — each asserts the intended
  post-fix semantics.

---

## Cross-stage interactions checked

- **C6-1 broadening + call boundary check at line 736**: the
  call boundary check explicitly skips `_compatible` when `pty`
  is TyVar / TySize / TyUnknown. So argument-vs-parameter checks
  at the call site are unchanged by the new top arm. The C6-1
  broadening affects only the FIVE other `_compatible` callsites
  (body / let / if-else / match-arm / pattern-bind).
- **C5-4 fix + check.py reach gap**: `check.py` doesn't call
  fn-mono at all. So `helixc check foo.hx` against a program
  with fn-mono `ShapeFoldError` still produces no diagnostic.
  This is a still-open cycle-5 deferred observation, not a new
  cycle-6 finding.
- **C5-3 TileLit walk + downstream codegen**: substituting
  let-bound names into TileLit shape preserves IntLit shape
  elements for downstream codegen. Surface reach: gated by
  Phase-0 tile-shape parser support; today user-reachable via
  stage 15+ surface syntax (`tile<f32, [N], REG>::zeros()` with
  N let-bound). The C5-3 finding's reachability gating is
  unchanged — the fix closes the SILENT-DROP hole regardless of
  reach.
- **monomorphize_safe docstring drift**: the wrapper's docstring
  still says "callers that don't care can ignore diags; the
  count is 0 in that case" — this is now stale because the only
  caller (x86_64.py) DOES care and aborts. Not a silent failure
  (no caller actually exercises the "don't care" path), but
  documentation drift that may mislead future contributors. Not
  flagged as a new finding because no silent failure mechanism
  exists; the harm surface is solely "future contributor adds
  a second caller that ignores diags and continues with
  half-mutated prog state". Noted as a deferred observation.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-7 candidates)

- **`monomorphize_safe` docstring drift**: the wrapper's
  docstring suggests callers MAY ignore diags. The only caller
  now aborts. A cycle-7 housekeeping batch could either rewrite
  the docstring ("callers MUST abort on non-empty diags") or
  rewrite `Monomorphizer.run` to catch per-instance (parallel
  to struct_mono's approach) so the wrapper's "(0, diags)"
  return reflects "0 SUCCESSFUL adds" and the prog state remains
  consistent. Either fix prevents future contributors from
  silently re-introducing the C5-4 hole.
- **check.py fn-mono reach**: `check.py` doesn't call
  `monomorphize_safe` at all. Helix programs with a fn-mono
  `ShapeFoldError` trigger via `helixc check foo.hx` see no
  diagnostic. Cycle-5-audit deferred. Cycle-6 didn't touch it.
  Could be addressed by adding `mono_count, mono_diags =
  monomorphize_safe(prog)` to check.py after struct_mono, with
  symmetric abort.
- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: the cycle-6
  commit message explicitly defers re-fix to a later cycle
  pending parse-time constant folding or a typecheck pass before
  closure capture. Stop-the-line on a CRITICAL is unresolved as
  of cycle 6.
- **F1 narrowing**: if cycle 7 adopts the per-shape-element
  narrowing recommendation in C6-1, the new arm should also
  cover `TyArray.size` AND `TyTensor.shape[i]` AND `TyTile.shape[i]`
  consistently (i.e., a single helper `_shape_compat(sa, sb)`
  used by all three structural arms).

---

## Summary

| #    | Severity | Location                                                    | Finding                                                                                                            |
|------|----------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| C6-1 | MEDIUM   | typecheck.py:2166-2179 + 1142 + 1164 + 1550 + 1576          | cycle-6 C5-2 / F1 top-level TyVar/TySize defer arm in `_compatible` broadens beyond the shape-recursion target → body/let/if-else/match-arm callsites silently accept generic-return-from-monomorphic-literal patterns |

**Total: 1 new finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).**

---

## Cycle 6 status

**Cycle 6 NOT clean.** Per the strict criterion (zero findings of
ANY severity), the 1 MEDIUM new finding BLOCKS the cycle-6 clean
determination.

### Stop-the-line determination: **NO**

C6-1 is MEDIUM — the broadening produces silent acceptance of
generic-return-from-monomorphic-literal patterns. These are
unusual in practice (most Helix code today doesn't write
`fn g[T]() -> T { 42 }`) and the harm surface is "user gets no
diagnostic for a likely type-pun bug", not a runtime miscompile.
The cycle-6 fix-sweep made strong progress: 4 of 4 cycle-5
findings closed at the mechanism level, with the only new
finding being a secondary effect of the chosen implementation
of F1.

C6-1's fix is mechanical (narrow F1 to per-shape-element defer
in the three structural arms; add 4 adjacent regression tests).
Recommend addressing in a cycle-7 fix batch alongside the
still-open carryover audit-C4-7 (check.py `except Exception`) and
the `monomorphize_safe` docstring drift.

### Cycle 6 → NEW FINDINGS COUNT for the strict-clean gate: 1 (0 CRITICAL + 0 HIGH + 1 MEDIUM + 0 LOW) — clean-counter remains at 0.

### Estimated remaining open findings going into cycle 7

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 1 of 6 still-open carryovers
  closed by cycle 6 (audit-C4-2 paper-only HIGH is now CLOSED
  via C5-2/F1 mechanism; audit-C4-3 root cause is also closed
  via the same arm). 4 still open: audit-C4-1 CRITICAL,
  audit-C4-4 HIGH, audit-C4-6 MEDIUM (superseded by C5-1 close
  → also closed), audit-C4-7 MEDIUM. Net: 3 still open.
- Cycle 4 type-design (sibling audit): partial close —
  E3 closed via C5-4/F3; E1 closed via C5-2/F1 mechanism;
  others unchanged.
- Cycle 4 codereview (sibling audit): 0 new (was already clean).
- Cycle 5 silent-failure: 4 new — all 4 CLOSED by cycle 6
  (C5-1 by revert; C5-2 by F1; C5-3 by F4; C5-4 by F3).
- Cycle 5 type-design + codereview (sibling audits, cycle-5
  parallel): F1..F6 from sibling audits all addressed in this
  cycle-6 fix-sweep; remaining cycle-5 carryovers are the
  cycle-5 deferred items.
- Cycle 6 silent-failure: 1 new (C6-1 open).
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 6 — cycle 6 didn't touch them).
- Cycle 6 net: 20 + 3 + (deferred type-design partial) + 1 = **≥24
  open findings** going into cycle 7. (Down from ≥34 at end of
  cycle 5 — cycle 6 closed roughly 10 findings net.)

Recommend prioritizing in this order for the cycle-7 fix batch:
1. **C6-1** (MEDIUM — narrow F1 to per-shape-element defer in
   the three structural arms; this is the only NEW cycle-6
   finding and the simplest mechanical fix).
2. **audit-C4-1** (CRITICAL — still-open from cycle 4; deferred
   in cycle 6 pending typechecker-before-closure-capture; the
   carryover deadline approaches as cycles 1-6 progress).
3. **audit-C4-4** (HIGH — D9 paper-only).
4. **audit-C4-7** (MEDIUM — narrow check.py `except Exception`
   to internal-error classes only).
5. **`monomorphize_safe` docstring drift** (LOW — rewrite
   docstring or refactor `Monomorphizer.run` per-instance).
6. **check.py fn-mono reach** (LOW — add `monomorphize_safe`
   call to `check.py` after struct_mono with symmetric abort).

After this batch lands, cycle 7 should re-audit. The "5 clean
cycles before Phase 0 deprecation" goal requires the strict
criterion (zero findings of any severity) to be met for
5 CONSECUTIVE cycles — cycle 6 is the 6th cycle and is NOT
clean, so the clean-counter remains at 0. Once the C6-1 fix
lands cleanly, cycle 7 can re-audit toward the first clean
cycle.
