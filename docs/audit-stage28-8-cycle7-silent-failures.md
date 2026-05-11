# Stage 28.8 Cycle 7 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: b8e047e (read-only audit). Cycle-7 fix-sweep range
c3f26ef..b8e047e (1 squashed fix-sweep commit covering C6-1 +
G1 + G2 closures).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically re-audits the cycle-7
fix-sweep changes for fresh silent windows introduced by the
fixes themselves.
**Trigger**: pre-Stage-29 audit gate — Cycle 7 of 5+ (the gate
re-arms each time a cycle is not clean). Re-audits same scope
after cycle-7 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Per the user directive for cycle 7,
findings already documented in cycles 1-6 are NOT re-flagged unless
they CHANGED in the cycle-7 fix-sweep.

**Method**:
1. Read prior cycle silent-failure docs (cycle 1 — 13 findings;
   cycle 2 — 6 findings; cycle 3 — 6 findings; cycle 4 — 8
   findings; cycle 5 — 4 findings; cycle 6 — 1 finding) to avoid
   re-flagging already-documented findings.
2. Walked `git show b8e047e` — the single cycle-7 fix-sweep
   commit. Read the diff for each of: typecheck.py
   (`_size_compatible` extraction + `_compatible` narrowing +
   TyMemTier G2 carve-out arms + G1 D-vs-Logic extra-text) and
   test_typecheck.py (rename of `test_c5_2_compatible_tysize_cascade`
   → `test_c5_2_size_compatible_tysize_cascade` + new
   `test_c6_1_compatible_tyvar_not_top_cascade`).
3. For each cycle-7 fix's diff, traced data flow forward to check
   whether the fix opened a fresh silent window, left a fix
   incomplete (paper-only), compounded a prior-cycle regression,
   or over-corrected.
4. Direct Python probes against `b8e047e` HEAD to confirm
   reproducer behavior at the body / let / if-else / match-arm
   `_compatible` callsites for the new narrowed cascade AND
   the new TyMemTier × TyVar/TySize carve-out arms (lines
   2208-2211).
5. Cross-checked the cycle-7 fix coverage against the still-open
   carryover findings from cycles 1-6 to identify which carryovers
   were actually CLOSED by cycle 7 vs which remain open.

**Result**: **1 new finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW)**
— Cycle 7 NOT clean. The fix-sweep makes EXCELLENT progress —
the cycle-6 C6-1 MEDIUM is correctly closed for the most-common
broad case (`_compatible(TyVar, TyPrim)` at body/let/if/match
now rejects again), via a clean `_size_compatible` extraction
that narrows the cycle-5 audit's option (b) into a dedicated
shape-position helper. The G1 LOW (D-vs-Logic extra-text
imprecision) is closed cleanly via a new `other_is_logic`
predicate.

The one new finding (C7-1) is a SECONDARY EFFECT of the cycle-7
G2 carve-out (new `_compatible` arms at lines 2208-2211 that
specifically accept `TyMemTier × (TyVar|TySize)` pairs as
deferred). The narrowing of C6-1 correctly tightens TyVar
behavior at value positions for TyPrim / TyTensor / TyArray /
TyTuple / etc. — but the explicit TyMemTier × TyVar/TySize
carve-out preserves the SAME body/let/if/match silent-acceptance
window that the cycle-6 C6-1 audit flagged, narrowed to the
TyMemTier-vs-generic pair only. The cycle-7 fix-sweep claims
"The body-vs-return value-position check correctly rejects again"
but this is true only when neither side is TyMemTier — for the
TyMemTier-vs-TyVar pair the body-vs-return check still silently
accepts. The cycle-7 sibling type-design audit accepted G2 as a
deliberate carve-out at the type-soundness layer (TyMemTier may
later mono to T), but from the silent-failure-audit lens this
asymmetry between value-position rejection and TyMemTier-position
deferral is undocumented at the four affected callsites and
unexercised by the regression suite.

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

### Finding C7-1: cycle-7 G2 `TyMemTier × (TyVar|TySize)` carve-out at `_compatible` (lines 2208-2211) silently accepts at body / let / if-else / match-arm callsites — asymmetric with the simultaneously-narrowed C6-1 close for TyPrim/TyTensor/TyArray/TyTuple, and undocumented at those four callsites

**Location**:
- helixc/frontend/typecheck.py:2208-2211 (the new TyMemTier ×
  (TyVar|TySize) carve-out arms, both orders)
- helixc/frontend/typecheck.py:1142 (`_check_fn_decl`:
  `_compatible(body_ty, sig.ret)`)
- helixc/frontend/typecheck.py:1164 (`_check_stmt`/Let:
  `_compatible(value_ty, declared)`)
- helixc/frontend/typecheck.py:1560 (`_check_expr`/If:
  `_compatible(t, e)` for if/else branches)
- helixc/frontend/typecheck.py:1586 (`_check_expr`/Match:
  `_compatible(first, t)` for match arms)
**Severity**: LOW
**Category**: cycle-7-fix-introduced silent-acceptance narrowing
incomplete (asymmetric vs C6-1's narrow close)
**Stage**: 28.8 cycle-7 commit b8e047e (G2 carve-out arms)

**Description**:
The cycle-7 fix-sweep replaced the cycle-6 F1 top-level TyVar /
TySize cascade in `_compatible` with two changes:

1. **Narrow**: a new `_size_compatible` helper (lines 2176-2190)
   routes the cascade-safe arm only for TyArray.size /
   TyTensor.shape / TyTile.shape compares — the cycle-5 audit's
   option (b). The top-level `_compatible` no longer auto-cascades
   for `TyVar` / `TySize`.

2. **Carve-out**: two new arms at lines 2208-2211 inside
   `_compatible` accept the `TyMemTier × (TyVar|TySize)` pair
   (both orders) BEFORE the broad TyMemTier strict-separation
   rejection at line 2212-2213:

   ```python
   if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
       return a.tier == b.tier and self._compatible(a.inner, b.inner)
   if isinstance(a, TyMemTier) and isinstance(b, (TyVar, TySize)):
       return True
   if isinstance(b, TyMemTier) and isinstance(a, (TyVar, TySize)):
       return True
   if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
       return False
   ```

The cycle-7 commit message claims "The body-vs-return value-
position check correctly rejects again." This is true for
`fn g[T]() -> T { 42 }` (body=TyPrim('i32'), ret=TyVar('T'))
because `_compatible(TyPrim, TyVar)` no longer has a top-level
cascade and falls through to `a == b` = False → rejects with
"body type i32 does not match return type T". The new test
`test_c6_1_compatible_tyvar_not_top_cascade` confirms this for
the TyPrim case.

But the body-vs-return claim is INCORRECT for the symmetric
`TyMemTier` pattern. For `fn h[T]() -> T { let m: WorkingMem<i32>
= ...; m }` (body=TyMemTier, ret=TyVar('T')), the new G2 carve-out
at line 2208 fires before the broad strict-separation rule and
returns True silently. Pre-cycle-6 the same compare returned
False (via the broad TyMemTier separation arm); cycle-6 returned
True via the broad cascade; cycle-7 returns True via the new
narrow G2 carve-out. From the silent-failure-audit lens, the
TyMemTier-vs-generic-T body-vs-return path remains silent across
the cycle-6 → cycle-7 transition, while the analogous TyPrim-vs-
generic-T path was just re-strictened.

Verified end-to-end via direct Python probe against b8e047e:

```python
>>> from helixc.frontend.typecheck import (
...     TypeChecker, TyVar, TyPrim, TyMemTier, TySize)
>>> from helixc.frontend import ast_nodes as A
>>> tc = TypeChecker(A.Program(module=None, items=[]))
>>> tc._compatible(TyMemTier(tier='WorkingMem',
...                          inner=TyPrim('i32')), TyVar('T'))
True   # silent — body=WorkingMem<i32>, ret=T accepted
>>> tc._compatible(TyVar('T'), TyPrim('i32'))
False  # correctly rejects (cycle-7 narrowing — closes C6-1)
>>> tc._compatible(TyMemTier(tier='WorkingMem',
...                          inner=TyPrim('i32')), TySize('N'))
True   # also silent for TyMemTier × TySize (kind-mismatch case)
```

The asymmetry affects the same five callsites that the cycle-6
C6-1 finding documented:

1. **Line 1142 — function body vs sig return**:
   `fn h[T]() -> T { let m: WorkingMem<i32> = ...; m }` silently
   typechecks. Analogous `fn g[T]() -> T { 42 }` correctly rejects
   post-cycle-7. The signal-vs-no-signal gap depends only on the
   body-type's kind (TyMemTier silent; everything else loud).

2. **Line 1164 — let with declared type**:
   `fn k[T]() -> T { let y: T = create_working_mem(); y }`
   silently accepts the `let y: T = WorkingMem<i32>` assignment.
   Analogous `let y: T = 42` correctly rejects.

3. **Line 1560 — if/else branches**:
   `fn l[T](b: bool, x: T) -> T { if b { x } else { create_wm() } }`
   silently picks the first arm's TyVar type and accepts the
   second arm's TyMemTier. Analogous `if x else 0` (T vs i32)
   correctly rejects.

4. **Line 1586 — match arm bodies**:
   `fn m[T](x: T) -> T { match 1 { 1 => x, _ => create_wm() } }`
   silently accepts mismatched arm types. Analogous match with
   i32 fallback correctly rejects.

5. **Pattern bindings** (transitively): same silent acceptance
   for any pattern bind that lands a TyMemTier value into a
   TyVar slot or vice versa.

The TyMemTier × TySize order (`_compatible(TyMemTier, TySize)`
also returns True via the symmetric arm at line 2211) is a
KIND-MISMATCH case: TySize is a count/dimension generic, not a
memory tier value. A TyMemTier could never legitimately mono to
a `[T; N]` size position. The G2 carve-out lumps TyVar and
TySize together symmetrically, but only the TyVar arm has the
"may later bind to TyMemTier" rationale; the TySize arm hides
a genuine kind error.

The cycle-7 sibling type-design audit (audit-stage28-8-cycle7-
type-design.md) accepted G2 as a deliberate "shape-position-only"
cascade design choice, but the silent-failure-audit lens looks at
the same code from the user's perspective: at the body / let /
if-else / match-arm callsites, the diagnostic-quality contract is
"if the user wrote `let y: T = create_wm()` with `T` a generic
param, they should at minimum see a warning that this is being
deferred to mono substitution." Currently they see nothing.

**Hidden errors**:
- `fn h[T]() -> T { let m: WorkingMem<i32> = ...; m }` silently
  typechecks. If the function is called with a binding `T = i32`
  the silent acceptance becomes a hard error at the call site
  ("expected i32, got WorkingMem<i32>"), but the user sees the
  error at the CALL not at the definition — confusing
  attribution.
- `fn k[T](b: bool, x: T) -> T { if b { x } else { create_wm() } }`
  silently picks the first arm's type and accepts the second
  arm's TyMemTier. The downstream usage of the result-of-if as
  a `T` value will surface confusingly when the second arm is
  reached at runtime — but the type system gave no signal at
  the definition site.
- `_compatible(TyMemTier, TySize)` returns True via the symmetric
  arm at line 2211. TySize is a count, not a value — this is
  a kind-mismatch with no legitimate mono substitution. The
  carve-out hides a genuine kind error that pre-cycle-6 would
  have rejected via the broad TyMemTier strict-separation rule.
- The cycle-7 fix-sweep does NOT add tests covering the TyMemTier
  side of the carve-out. `test_c6_1_compatible_tyvar_not_top_
  cascade` asserts `_compatible(TyVar, TyPrim)` rejects but not
  `_compatible(TyMemTier, TyVar)` (which post-fix returns True).
- A future contributor reading the new G2 carve-out comment
  ("Audit 28.8 cycle 7 G2 carve-out: TyVar / TySize on either
  side is NOT considered cross-tier") may not realize that the
  carve-out applies at ALL callsites (body, let, if, match) not
  just at shape positions — the cycle-7 narrowing's primary
  intent was to confine cascade-safe behavior to shape positions
  via `_size_compatible`, but G2 re-introduces a per-kind cascade
  exception at the top level. The carve-out's reach is broader
  than the comment suggests.
- The cycle-7 commit message ("The body-vs-return value-position
  check correctly rejects again") creates a documentation drift
  — true for most types, false for TyMemTier.

**Recommendation**:
1. Move the G2 carve-out OUT of `_compatible` and into
   `_size_compatible`, parallel to the cycle-7 C6-1 narrowing.
   Body / let / if-else / match-arm value positions use the full
   `_compatible`; size positions use `_size_compatible`. The
   TyMemTier × TyVar/TySize defer becomes a shape-position-only
   rule, symmetric with the rest of the cycle-7 narrowing.

   Specifically: remove lines 2208-2211 from `_compatible`. Add
   the TyMemTier × TyVar/TySize defer to `_size_compatible`
   (already handled by the first-arm TyVar/TySize check at line
   2184, so no new code needed — just remove the top-level G2
   carve-out).

2. Alternatively (if the G2 type-design rationale is preferred):
   add a per-callsite check at lines 1142, 1164, 1560, 1586 that
   emits an explicit "warning: TyMemTier × generic-T deferred to
   mono substitution" diagnostic when the asymmetric pair is
   detected. Users get a signal that the type check was relaxed,
   without losing the underlying mono-substitution flexibility.

3. Drop the TySize half of the G2 carve-out (line 2211's
   `isinstance(a, (TyVar, TySize))` clause). TySize is a kind
   distinct from TyMemTier's value kind; a generic size param
   could not legitimately mono to a memory-tier value. Narrow
   the carve-out to `(TyVar)` only.

4. Add regression tests covering:
   - `fn h[T]() -> T { let m: WorkingMem<i32> = ...; m }` must
     either error (preferred per silent-failure lens) or emit a
     deferred-to-mono warning.
   - `_compatible(TyMemTier(...), TySize('N'))` must error.
   - `_compatible(TyMemTier(WorkingMem<i32>), TyMemTier(LongTermMem<i32>))`
     still rejects (verify the carve-out doesn't accidentally
     loosen the strict-separation rule when both sides are
     TyMemTier).

5. Update the cycle-7 commit message's "The body-vs-return
   value-position check correctly rejects again" claim to specify
   the TyMemTier carve-out exception, OR amend the G2 carve-out
   to make the claim universally true (per recommendations 1-3).

**Trap-id**: n/a (typecheck silent-accept, no trap-id).

---

## Cycle 7 fix-sweep re-verification

Each cycle-7 fix-sweep change was inspected for paper-only fixes,
silent windows, false positives, and state-leak. The cycle-7
fix-sweep landed as a single commit (b8e047e) covering: C6-1
narrowing via `_size_compatible` extraction + G1 D-vs-Logic
extra-text + G2 TyMemTier × TyVar/TySize carve-out.

| fix-sweep label | What changed | Audit-doc cross-ref | C7 verdict |
|---|---|---|---|
| C6-1 narrow | New `_size_compatible` helper; top-level `_compatible` no longer auto-cascades TyVar/TySize; TyArray/TyTensor/TyTile size-element compares route through `_size_compatible` | C6-1 (cycle-6 silent-failure MEDIUM) | **closed at value position for TyPrim/TyTensor/TyArray/TyTuple**; partially open for TyMemTier (see C7-1 below) — the narrowed `_compatible` correctly rejects `_compatible(TyVar('T'), TyPrim('i32'))` per the new `test_c6_1_compatible_tyvar_not_top_cascade`, restoring the pre-cycle-6 body-vs-return diagnostic for the common case |
| G1 (D-vs-Logic text) | New `other_is_logic` predicate in D-arm extra-text branch; says "(one side D-wrapped, other Logic-wrapped)" when the other side is Logic-wrapped, else "(one side D-wrapped, other bare)" | G1 (cycle-6 type-design LOW) | **closed for D-vs-Logic**; deferred observation for D-vs-Quote (still says "bare" for Quote-wrapped other side — not flagged because cycle-6 G1 didn't flag the Quote case either; cycle-7 did not change Quote-vs-D behavior) |
| G2 carve-out | New `_compatible` arms `TyMemTier × (TyVar|TySize)` (both orders) returning True before the broad strict-separation rule | G2 (cycle-6 type-design LOW) | **opens C7-1** — the carve-out applies at top-level `_compatible`, so it fires at body / let / if-else / match-arm callsites as well as the call-boundary check. From the silent-failure-audit lens this is an asymmetric narrowing of C6-1 |
| test rename + add | `test_c5_2_compatible_tysize_cascade` → `test_c5_2_size_compatible_tysize_cascade` (semantic rename — the helper changed); new `test_c6_1_compatible_tyvar_not_top_cascade` for the narrow-vs-broad assertion | (housekeeping) | OK — rename matches the helper extraction; new test asserts the intended post-fix rejection for the TyVar/TyPrim pair. Coverage gap: no test asserts `_compatible(TyMemTier, TyVar)` returns True per the new G2 carve-out, nor that the TyMemTier-body-T-ret case is silently accepted — see C7-1 recommendation 4 |

### Specific re-verifications from the audit instructions

- **`_size_compatible` extraction**: verified via direct file
  inspection at lines 2176-2190. The helper's four arms run in
  order: TyVar/TySize (cascade-pass), TyUnknown (cascade-pass),
  identity (a == b), then delegate to `_compatible(a, b)` for
  the genuine non-cascade structural compare. The TyArray size
  compare at line 2251 (`a.size == b.size or self._size_compatible(
  a.size, b.size)`) preserves the identity-fastpath then falls
  to the helper. TyTensor.shape and TyTile.shape compares at
  lines 2283 and 2293 zip-compare via `_size_compatible`. None
  of the three callsites route value-position types through
  `_size_compatible` accidentally.
- **`_compatible` top-level cascade removal**: Probed via Python
  repro at b8e047e — `_compatible(TyPrim('i32'), TyVar('T'))`
  returns False (cycle-6 returned True). `_compatible(TyVar('T'),
  TyVar('U'))` returns False (cycle-6 returned True; cycle-7
  same-name TyVar still passes via `a == b`). `_compatible(
  TyTensor(...), TyVar('T'))` returns False (cycle-6 True;
  cycle-7 False because the TyTensor structural arm doesn't
  match when one side is TyVar). All match the intended
  narrowing.
- **G2 carve-out (lines 2208-2211)**: Probed via Python repro —
  `_compatible(TyMemTier('WorkingMem', TyPrim('i32')), TyVar('T'))`
  returns True (silently). `_compatible(TyMemTier('WorkingMem',
  TyPrim('i32')), TySize('N'))` returns True (silently — kind
  mismatch). `_compatible(TyMemTier('WorkingMem', TyPrim('i32')),
  TyMemTier('LongTermMem', TyPrim('i32')))` returns False
  (strict-separation preserved — both-MemTier arm runs first
  and rejects on tier mismatch). The carve-out is correctly
  scoped to the asymmetric case but applies at all `_compatible`
  callsites, not just shape positions. See C7-1.
- **G1 D-vs-Logic extra-text**: Probed by walking the eight-row
  matrix from the cycle-6 G1 finding through the new
  `other_is_logic` predicate (lines 1371-1378). All eight rows
  produce correct text per the cycle-7 type-design audit's
  re-verification. No silent-failure surface introduced. The
  D-vs-Quote case (l=Quote<T>, r=D<T>) still emits
  "(one side D-wrapped, other bare)" — imprecise but not a
  CHANGED behavior (cycle-6 G1 didn't flag the Quote case);
  noted as deferred observation, not a new finding.
- **test rename + add**: `test_c5_2_size_compatible_tysize_cascade`
  correctly probes the new helper; `test_c6_1_compatible_tyvar_not_
  top_cascade` correctly probes the rejection. Coverage gap:
  no assertion that `_compatible(TyMemTier, TyVar)` returns True
  per G2 — the silent-acceptance window is unexercised.

### Carryover findings status (cycles 1-6)

The cycle-7 fix-sweep did NOT re-attempt the following
still-open carryover findings (acknowledged in the cycle-6
audit's "Estimated remaining open findings going into cycle 7"
section):

| Carryover | Severity | Cycle-7 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — cycle-7 did not address; deferred per cycle-6's commit message pending parse-time constant folding or a typecheck pass before closure capture |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-7 (check.py `except Exception`) | MEDIUM | **still open** — not addressed |
| audit-C4-8 deferred (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| monomorphize_safe docstring drift (cycle-6 deferred) | (not a finding) | **still open** — docstring still suggests callers MAY ignore diags; only caller now aborts |

These are NOT re-flagged as new cycle-7 findings per the user
directive (already documented in cycles 1-6, did not CHANGE in
cycle 7's fix-sweep). They remain in the open-findings ledger.

### Specific items checked clean in cycle 7 (no new finding)

- The cycle-6 F1 top-level TyVar/TySize cascade is genuinely
  removed from `_compatible`. The narrowed `_size_compatible`
  helper carries the cascade-safe behavior only at TyArray /
  TyTensor / TyTile size positions. The call-boundary check at
  line 736 still skips top-level TyVar/TySize at `pty` BEFORE
  invoking `_compatible`, so the pre-cycle-6 generic-arg-at-
  value-position path is preserved.
- The new `_size_compatible` helper's structural delegation to
  `self._compatible(a, b)` for non-cascade cases is correct —
  weird types in size positions (e.g., TyTensor in size) get
  the strict structural compare. No silent acceptance introduced
  at size positions.
- The G1 D-vs-Logic predicate (line 1371-1378) correctly
  distinguishes the four wrap-asymmetry rows for D ∈ {bare, D,
  Logic, D<Logic>} on each side. The cycle-6 G1 inaccurate text
  is fixed for all relevant rows.
- The test rename `test_c5_2_compatible_tysize_cascade` →
  `test_c5_2_size_compatible_tysize_cascade` matches the helper
  extraction's semantic change; the renamed test correctly
  probes the new helper rather than the removed top-level arm.
- Both-MemTier compare (line 2206-2207) still requires
  `a.tier == b.tier` AND inner-compatible. Strict separation
  between WorkingMem / LongTermMem / etc. is preserved.

---

## Cross-stage interactions checked

- **C7-1 carve-out reach + call boundary check at line 736**:
  the call boundary check explicitly skips `_compatible` when
  `pty` is TyVar / TySize / TyUnknown. So argument-vs-parameter
  checks at the call site are unchanged by the new G2 arms when
  `pty` is TyVar. But when `pty` is TyMemTier and `aty` is
  TyVar (a generic-T value passed where TyMemTier is expected),
  the boundary check enters `_compatible(TyMemTier, TyVar)` which
  now returns True via G2 — same silent path as the body-vs-
  return case. The call-boundary check thus also defers
  TyMemTier × TyVar pairs to mono, which may be the intended
  behavior but is uncovered by tests.
- **G2 carve-out + `_size_compatible` recursion**: the helper
  at line 2190 delegates to `self._compatible(a, b)` for non-
  cascade cases. If a TyMemTier ever lands in a shape position
  (e.g., a malformed TyArray with size=TyMemTier) the G2 arm
  would fire if the other side is TyVar/TySize, otherwise the
  broad strict-separation arm rejects. No silent failure at
  shape positions specifically.
- **G1 D-vs-Logic + D-vs-Quote**: the Quote-wrapped case
  (l=Quote<T>, r=D<T>) still emits "(one side D-wrapped, other
  bare)" — cycle-7 G1 fix only handles the Logic case. This is
  not a CHANGED behavior (cycle-6 G1 also didn't flag Quote);
  noted as deferred observation. A complete fix would generalize
  the `other_is_logic` predicate to `other_wrapper_kind` returning
  "Logic-wrapped" / "Quote-wrapped" / "bare" as appropriate.
- **C7-1 + TyVar same-name pass**: `_compatible(TyVar('T'),
  TyVar('T'))` still returns True via `a == b` (TyVar dataclass
  equality on `name`). Same-name TyVar pass is preserved (no
  cycle-6/7 regression on `fn id[T](x: T) -> T { x }`).

---

## Deferred / out-of-scope observations (NOT new findings; cycle-8 candidates)

- **G2 carve-out scope**: if cycle 8 adopts the C7-1
  recommendation 1 (move the carve-out into `_size_compatible`),
  the type-design G2 finding may resurface as a question — does
  the TyMemTier strict-separation contract apply at shape
  positions too? Today the carve-out is at top-level `_compatible`,
  so it also defers at shape positions; moving it to
  `_size_compatible` only would preserve shape-position deferral
  symmetric with the rest of the cycle-7 narrowing. The
  cycle-7-type-design audit's G2 close presumed the carve-out
  applies at the call boundary; that presumption holds for the
  shape-position-only variant.
- **D-vs-Quote diagnostic text**: the cycle-7 G1 fix covers
  Logic but not Quote. A cycle-8 housekeeping batch could
  generalize the predicate to `_other_wrapper_kind(l_is_diff,
  r_is_diff, l_is_logic, r_is_logic, l_is_quote, r_is_quote)`
  returning the appropriate kind name. Not flagged because
  cycle-6 G1 didn't include the Quote case.
- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still open
  CRITICAL. Cycle-7 did not address; the cycle-6 commit message's
  deferral rationale (pending parse-time constant folding or a
  typecheck pass before closure capture) still applies.
- **monomorphize_safe docstring drift**: still open (cycle-6
  deferred). Docstring suggests callers MAY ignore diags; the
  only caller (x86_64.py) now aborts. Could be addressed by
  rewriting the docstring or refactoring `Monomorphizer.run`
  per-instance.
- **check.py fn-mono reach**: still open (cycle-5 deferred).
  `check.py` doesn't call `monomorphize_safe` at all.
- **C7-1 narrowing path**: if cycle 8 adopts C7-1 recommendation
  2 instead (per-callsite deferred-to-mono warning), the four
  callsites (1142, 1164, 1560, 1586) should share a helper
  `_emit_generic_defer_warning(span, lhs, rhs, position_name)`
  to avoid copy-paste drift.

---

## Summary

| #    | Severity | Location                                                    | Finding                                                                                                            |
|------|----------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| C7-1 | LOW      | typecheck.py:2208-2211 + 1142 + 1164 + 1560 + 1586          | cycle-7 G2 TyMemTier × (TyVar|TySize) carve-out at top-level `_compatible` silently accepts at body/let/if-else/match-arm callsites — asymmetric with the simultaneously-narrowed C6-1 close for TyPrim/TyTensor/TyArray/TyTuple, and undocumented at those four callsites |

**Total: 1 new finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).**

---

## Cycle 7 status

**Cycle 7 NOT clean.** Per the strict criterion (zero findings of
ANY severity), the 1 LOW new finding BLOCKS the cycle-7 clean
determination.

### Stop-the-line determination: **NO**

C7-1 is LOW — the silent acceptance is narrow (specifically
TyMemTier × TyVar/TySize pairs at value positions), and the
deferred behavior may be recovered at the call boundary when
the generic is mono'd. The cycle-7 fix-sweep made strong
progress: the C6-1 MEDIUM is closed for the dominant case
(TyPrim/TyTensor/TyArray/TyTuple at value positions), the G1
LOW is closed cleanly, and the G2 LOW is closed at the type-
soundness layer. The only new finding is an asymmetric remnant
of C6-1 narrowed to the TyMemTier-vs-generic-T specific pair.

C7-1's fix is mechanical (move the G2 carve-out into
`_size_compatible` per recommendation 1; OR emit a per-callsite
deferred-to-mono warning per recommendation 2; OR drop the TySize
half of the carve-out per recommendation 3; add 3 adjacent
regression tests per recommendation 4). Recommend addressing in
a cycle-8 fix batch alongside the still-open carryover
audit-C4-1 (CRITICAL) and audit-C4-7 (MEDIUM).

### Cycle 7 → NEW FINDINGS COUNT for the strict-clean gate: 1 (0 CRITICAL + 0 HIGH + 0 MEDIUM + 1 LOW) — clean-counter remains at 0.

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW). ← here

Trend: monotonic severity decrease cycle-over-cycle for the
last four cycles. Cycle 7 is the first cycle since cycle 1
where no finding is MEDIUM or higher. Cycle 8 has a credible
chance of being the first clean cycle.

### Estimated remaining open findings going into cycle 8

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 5 closed by cycles 5-7. 3
  still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH, audit-C4-7
  MEDIUM. Net: 3 still open.
- Cycle 4 type-design (sibling audit): partial close — E3
  closed via C5-4/F3; E1 closed via C5-2/F1 mechanism; others
  unchanged.
- Cycle 4 codereview (sibling audit): 0 new (was already clean).
- Cycle 5 silent-failure: 4 new — all 4 CLOSED by cycle 6.
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED for value
  positions (TyPrim/TyTensor/TyArray/TyTuple) by cycle 7;
  partial-narrow open for TyMemTier × TyVar/TySize via C7-1.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED by cycle 7
  at the type-soundness layer; G2's silent-failure-audit-lens
  reading opens C7-1.
- Cycle 7 silent-failure: 1 new (C7-1 open).
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 7 — cycle 7 didn't touch them).
- Cycle 7 net: 20 + 3 + (deferred type-design partial) + 1 =
  **≥24 open findings** going into cycle 8. (Roughly unchanged
  from end of cycle 6 — cycle 7 closed C6-1 and 2 type-design
  LOWs but opened C7-1. Net delta: -2 closed.)

Recommend prioritizing in this order for the cycle-8 fix batch:
1. **C7-1** (LOW — move G2 carve-out into `_size_compatible` per
   recommendation 1, or emit per-callsite deferred-to-mono
   warning per recommendation 2; this is the only NEW cycle-7
   finding and the simplest mechanical fix).
2. **audit-C4-1** (CRITICAL — still-open from cycle 4; deferred
   in cycles 6-7 pending typechecker-before-closure-capture;
   the carryover deadline approaches as cycles 1-7 progress).
3. **audit-C4-4** (HIGH — D9 paper-only).
4. **audit-C4-7** (MEDIUM — narrow check.py `except Exception`
   to internal-error classes only).
5. **monomorphize_safe docstring drift** (housekeeping — rewrite
   docstring or refactor `Monomorphizer.run` per-instance).
6. **check.py fn-mono reach** (housekeeping — add
   `monomorphize_safe` call to `check.py` after struct_mono with
   symmetric abort).
7. **D-vs-Quote diagnostic text** (housekeeping — generalize the
   `other_is_logic` predicate from cycle-7 G1 to also cover
   Quote-wrapped other side).

After this batch lands, cycle 8 should re-audit. The "5 clean
cycles before Phase 0 deprecation" goal requires the strict
criterion (zero findings of any severity) to be met for
5 CONSECUTIVE cycles — cycle 7 is the 7th cycle and is NOT
clean, so the clean-counter remains at 0. Cycle 7 is the
closest any cycle has come to clean (1 LOW, no MEDIUM+), so
cycle 8 has the strongest probability yet of being the first
clean cycle.
