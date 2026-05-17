# Stage 48 Inc 4 — Gate-4 Silent-Failure Audit

**Date:** 2026-05-17
**HEAD:** `3415727` ("Stage 48 Inc 4 closure gate-3 G3-F1 scope-aware
fix + M1/M2/L1 polish")
**Lens:** silent failures (gate-4 of 3-clean-gate closure)
**Streak counter at start:** 0/3 (gates 1+2+3 each found one HIGH).

> Per the cascading-defect rhythm from gates 1-3 of this stage, the
> auditor enters gate-4 *expecting* to find one more HIGH or close
> the streak. The scope is the gate-3 patch (`c32dfbb..3415727`) plus
> the full Stage 48 diff (`48db12d..3415727`) re-verified end-to-end.

---

## Scope

Read-only audit over:

- `helixc/frontend/typecheck.py` — `_check_question`,
  `_check_block` snapshot/restore (with the gate-3 scope-aware
  set-stack), `_check_fn` entry-clear (M5), Assign-arm provenance
  invalidation (Stage 46 G3-F1 follow-up), Let-stmt provenance write
  + let-set bookkeeping, non-Result-operand diagnostic
  (M1-named-operand).
- `helixc/ir/lower_ast.py` — `__try` identity-tuple,
  `_lower_type` Result arm, both carrying `TODO(stage49)` markers.
- `helixc/tests/test_stage48_try.py` — 18 tests; runtime assertions
  via WSL ELF execution for the 4 happy-path/IR tests, typecheck-only
  assertions for the 14 rejection/provenance tests.

Diff stats:

| Patch              | Files | +Lines | -Lines |
|--------------------|-------|--------|--------|
| Stage 48 full      | 5     | 788    | 28     |
| Gate-3 only        | 4     | 294    | 41     |

---

## Defect-pattern checklist (12 patterns)

Each pattern from the gate-4 brief was end-to-end probed (parse →
typecheck → IR-lower → x86_64 ELF → WSL run + exit code).

| # | Pattern                                          | Result        |
|---|--------------------------------------------------|---------------|
| 1 | Wrong-arm provenance (gate-1 F2)                 | regression-free |
| 2 | Cross-fn carry (gate-2 M5)                       | regression-free |
| 3 | Inner-LET shadow (gate-2 F1)                     | regression-free |
| 4 | Inner-ASSIGN mutation (gate-3 G3-F1)             | regression-free |
| 5 | Nested-block cascading restore                   | composes correctly |
| 6 | Match-arm Block body                             | gate-3 G3-F1c covers |
| 7 | If/else-arm Block body                           | gate-3 G3-F1b covers |
| 8 | Bare-expression match-arm Assign body            | **HIGH G4-F1**  |
| 9 | ASSIGN-then-LET-shadow on same name              | **HIGH G4-F2**  |
| 10| Param `r` cross-fn false-reject (gate-2 M2/M5)   | regression-free |
| 11| `_check_fn` entry-clear for nested fn defs       | N/A — Helix has no nested fns |
| 12| Loop-body `?` (while/for)                        | provenance reaches inner block |
| 13| 3-deep `?` chain                                 | works end-to-end (exit 42) |
| 14| Exception-safety in `_pop_local_const_scope`     | gate-3 CR-M1 fix verified |

Detailed findings below.

---

## Findings

### G4-F1 — HIGH (confidence 95): match-arm bare-Assign body bypasses scope-restore, leading to silent miscompile via stale provenance

**Location:** `helixc/frontend/typecheck.py:4952-4978`
(`_check_expr` Match arm — `arm_tys.append(self._check_expr(arm.body, inner))`).

**Defect class:** Same family as gate-2 F1 + gate-3 G3-F1
(scope-bound provenance leak), surfaced through a different
syntactic vehicle: a match arm body that is a bare expression
(not a Block) does NOT enter `_check_block`, so the
`_result_constructor_provenance` snapshot/restore never fires
for that arm. Any `Assign` mutation inside the arm body leaks
into the surrounding scope's dict.

**Minimal repro (end-to-end exit code 99 verified on WSL):**

```helix
fn helper(x: i32) -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(7);
    match x {
        0 => r = Err(99),       // bare-Assign arm body — bypasses _check_block
        _ => r = Ok(7),          // bare-Assign arm body — bypasses _check_block
    };
    let v: i32 = r?;             // dict says r='ok' (last arm won), accept; runtime r=Err(99)
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper(0)) }
```

Sequence:
1. Outer fn block: dict `{r:'ok'}` after the let.
2. Match dispatch: each arm's body is checked via `_check_expr`
   (line 4966), not `_check_block`. Both bodies are bare
   `Assign` expressions → the Assign-arm (line 5003-5035) fires
   for each, updating the dict sequentially. Last arm wins:
   dict `{r:'ok'}`.
3. After match: dict `{r:'ok'}`. The `?` provenance check sees
   `'ok'` and accepts (benign).
4. Identity-lower runs. At runtime with `x=0`, arm 0 fires:
   `r` slot becomes Err(99). `r?` extracts the Err bits as
   the Ok inner → process exits 99.

**Hidden errors:** the typecheck dict is structurally unable to
model "two arms with different constructor mutations" through
the bare-expr path because there's no scope vehicle to snapshot
into. The flat-dict per-name design assumes mutations are
sequenced; branching control flow is unmodeled.

**User impact:** any user writing concise match arms (`pat =>
r = ...`) without a Block wrapper gets a silent miscompile at
runtime. The `?` diagnostic is the user's primary safety net
in Phase-0; this case escapes it.

**Concrete fix sketch:** wrap match-arm-body and if/else-arm-body
checks in a single-use `_check_block`-equivalent snapshot/restore
helper, *or* refactor the arm-body call site to push/pop the
provenance map and let-scope set inline. Specifically at
typecheck.py:4966 and at the If-else arm (line 4944):

```python
# Suggested helper (mirrors _check_block's snapshot/restore):
def _check_expr_in_block_scope(self, expr, scope):
    saved = dict(self._result_constructor_provenance)
    self._result_let_block_scopes.append(set())
    try:
        return self._check_expr(expr, scope)
    finally:
        inner_lets = self._result_let_block_scopes.pop()
        mutated = {
            n for n in saved
            if n not in inner_lets
            and self._result_constructor_provenance.get(n) != saved.get(n)
        }
        self._result_constructor_provenance = saved
        for n in mutated:
            self._result_constructor_provenance.pop(n, None)
```

Then replace `self._check_expr(arm.body, inner)` (line 4966)
and the bare-expr else_ branch (line 4944) with the helper.
Same fix applies at line 4959 (guard expressions — paranoid
defence; guards are typically pure but an Assign inside a guard
would leak otherwise).

**Confidence:** 95.

---

### G4-F2 — HIGH (confidence 92): inner-block ASSIGN-then-LET-shadow on same name lets stale outer provenance survive restore, silent miscompile

**Location:** `helixc/frontend/typecheck.py:2451-2462` (the gate-3
G3-F1 mutated-name detection in `_check_block`'s finally block),
interacting with `helixc/frontend/typecheck.py:2517-2518`
(Let-stmt unconditionally adds the bound name to the inner
let-set).

**Defect class:** the gate-3 G3-F1 fix uses the let-set as a
"don't check this name for outer-mutation" mask. The mask is
applied PER NAME, not PER EVENT. A block that does (a) ASSIGN
to outer `r`, then (b) shadow-LET a new `r`, ends up with the
name `r` in `inner_lets` — which then SKIPS the mutation check
for the prior ASSIGN. The outer's stale provenance is restored
unchanged.

**Minimal repro (end-to-end exit code 99 verified on WSL):**

```helix
fn helper() -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(11);
    {
        r = Err(99);                         // ASSIGN to outer r (dict now {r:'err'})
        let r: Result<i32, i32> = Ok(5);     // inner-let shadow (inner_lets={r}, dict {r:'ok'})
        let _x: i32 = unwrap_ok(r);
    };
    let v: i32 = r?;                          // restore: r in inner_lets → SKIPPED → dict={r:'ok'} (saved); accept!
    Ok(v)                                     // runtime: outer r=Err(99) → identity-lower → exit 99
}
fn main() -> i32 { unwrap_ok(helper()) }
```

Trace through the restore (lines 2451-2462):
- `saved_provenance = {r:'ok'}` (snapshot at block entry)
- After `r = Err(99)`: dict `{r:'err'}` (Assign-arm wrote)
- After `let r = Ok(5)`: dict `{r:'ok'}` (Let-stmt's prov-set
  wrote at line 2533-2536, AND `inner_lets={r}` at line 2518)
- Restore: iterate saved (`{r}`); for r, `r in inner_lets`
  → SKIP mutation check → `mutated_outer_names = {}`
- `self.prov = saved = {r:'ok'}`. Pop nothing.

The outer `r?` then sees `prov[r]='ok'` and accepts. Runtime
r holds Err(99) from the inner ASSIGN. Identity-lower yields
the Err bits as the Ok inner → exit 99.

**Hidden errors:** any code path that writes-via-assign-then-
shadows is silently miscompiled. This pattern is plausible in
real code (early-exit guard reassigning `r` to a sentinel,
then re-binding `r` to a local computation). The dict's "last
write wins" model collides with the let-set's "name was
introduced inside this scope" semantics.

**Note on gate-2 vs gate-3 attribution:** pre-gate-3 (gate-2
simple snapshot/restore) ALSO mis-compiled this exact source
— the Let-stmt's prov-set writes 'ok', restore puts back saved
`{r:'ok'}`, same outcome. So strictly speaking this is a
**pre-existing F1-class flat-dict architectural limitation**
that gate-3 did NOT eliminate even though its scope-aware
design intended to. It is documented here as gate-4's new HIGH
because gate-3 explicitly claimed to "scope-disambiguate WHO
caused any post-block diff" — the disambiguation is incomplete:
it disambiguates LETs from ASSIGNs but not "LET-after-ASSIGN
on the same name" from "LET-only".

**User impact:** identical to G4-F1 — silent runtime miscompile,
no diagnostic.

**Concrete fix sketch:** track ASSIGN events separately from
LET events. Either:

1. Add a parallel `_result_assigns_block_scopes: list[set[str]]`
   stack populated by the Assign-arm (line 5021-5035) and
   consulted alongside `_result_let_block_scopes` at restore:

   ```python
   inner_assigns = self._result_assigns_block_scopes.pop()
   mutated_outer_names = {
       n for n in saved_provenance
       if (n in inner_assigns       # was assigned at least once
           or (n not in inner_lets
               and self._result_constructor_provenance.get(n)
                   != saved_provenance.get(n)))
   }
   ```

   Then a name that was BOTH assigned and let-shadowed in the
   block is still treated as mutated → the outer's stale entry
   drops, and the `?` falls through to F1-dynamic (typecheck-
   clean, runtime exit 99 acknowledged as F1-class — strictly
   better than silently accepting under a stale 'ok' claim).

2. OR (more conservative): at Let-stmt time, if the name is
   already in the current scope's let-set or assign-set, treat
   it as a re-binding event and propagate the mutation flag.

Fix (1) is the minimal sound change and aligns with the
"closing the runtime miscompile by dropping the static claim"
philosophy that gate-3 G3-F1 established.

**Confidence:** 92.

---

### G4-M3 — MEDIUM (confidence 75): match-arm guard expressions skip provenance scope-restore

**Location:** `helixc/frontend/typecheck.py:4958-4965`
(`_check_expr(arm.guard, inner)`).

**Defect class:** same scope-bypass family as G4-F1. Match
guards are checked via `_check_expr` (no `_check_block`
wrapper). A guard expression that contains an `Assign`
(`if r = compute_and_check()` — admittedly contrived in Helix
where `=` is statement-style, but the parser will accept it
inside an expression context with `expr.op == "="`) would leak.

**Minimal repro (not exit-99 verified — Helix's guard
grammar may not currently permit it; flagged as paranoid-
defence MEDIUM):**

```helix
fn helper(x: i32) -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(7);
    let dummy: i32 = match x {
        v if { r = Err(99); v > 0 } => 1,
        _ => 2,
    };
    let v: i32 = r?;
    Ok(v)
}
```

**User impact:** low frequency (guards are typically pure), but
the same scope-bypass class as G4-F1 — if G4-F1 is fixed via
the inline-helper approach, this site should also be wrapped.

**Concrete fix sketch:** same `_check_expr_in_block_scope`
helper as G4-F1.

**Confidence:** 75 (need to verify Helix's guard grammar
admits an inner Assign; the principle is sound regardless).

---

### G4-L4 — LOW (confidence 80): `dict()` snapshot + `_result_let_block_scopes.append()` precede the try-finally, leaking const-scope on (extremely-unlikely) exception

**Location:** `helixc/frontend/typecheck.py:2400-2402`.

```python
saved_provenance = dict(self._result_constructor_provenance)  # line 2400
self._result_let_block_scopes.append(set())                    # line 2401
try:                                                            # line 2402
    ...
finally:
    try:
        self._pop_local_const_scope()
    finally:
        ...
```

`_push_local_const_scope()` happened at line 2380. If `dict()`
or `.append()` raise (memory error, etc.), the const-scope
push is leaked — no finally fires. Gate-3 CR-M1 fixed the
inverse (`_pop_local_const_scope` raising mid-finally), but
the symmetric pre-try hazard remains.

**User impact:** practically nil — `dict()` and `set().append()`
don't raise under realistic memory conditions. Flagged for
discipline parity with CR-M1.

**Concrete fix sketch:** move `_push_local_const_scope`,
`dict()`, and `.append()` inside the try, with corresponding
cleanup in finally guarded on whether the push happened.
Stage 49 work — not a closure blocker.

**Confidence:** 80 (the hazard is real but unreachable in
practice).

---

### G4-L5 — LOW (confidence 70): `_check_fn` clears `_result_constructor_provenance` but not `_result_let_block_scopes`

**Location:** `helixc/frontend/typecheck.py:2274` (M5 fix).

The M5 fix at fn entry resets the provenance dict. The
parallel let-scope stack is NOT reset. In normal flow this
is harmless — `_check_block`'s push/pop is balanced. But if
a prior fn's `_check_block` raised mid-body (between the
`.append(set())` at line 2401 and the finally's `.pop()` at
line 2451), the stack would carry stale entries into the next
fn.

The fn-pass at typecheck.py:760-764 catches `TypeError_` (a
typed exception) but not generic exceptions. A generic
exception escaping `_check_block` between push and pop is
the failure window.

**User impact:** zero in practice for typecheck (generic
exceptions are bugs, not user-facing). Flagged for
defensive-symmetry parity with the M5 fix.

**Concrete fix sketch:** clear `self._result_let_block_scopes
= []` alongside the provenance dict at line 2274. One line.

**Confidence:** 70.

---

## Patterns confirmed clean

| Pattern                                              | Verification |
|------------------------------------------------------|--------------|
| `?` on typed-let Err (`let r: Result = Err(99); r?`) | rejected (gate-1 F2 regression test passes) |
| Cross-fn provenance carry (param `r` after fn-A `let r = Ok`) | not false-rejected (gate-2 M5) |
| Inner-block-LET shadow (`{ let r = Ok(5); }` then outer `r?`) | outer Err provenance survives (gate-2 F1) |
| Inner-block-ASSIGN-only mutation (`{ r = Err(99); }` then outer `r?`) | drops to F1-dynamic, no false static claim (gate-3 G3-F1) |
| If-then-arm Block-body assign mirror                 | gate-3 G3-F1b passes |
| Match-arm Block-body assign mirror                   | gate-3 G3-F1c passes |
| Nested-block cascading restore composition           | 2-deep + 3-deep probes both rejected/accepted correctly per branch |
| 3-deep `?` chain through Ok-returning fns            | exit 42 end-to-end |
| While-body `?` on Err-constructed name               | rejected at typecheck (provenance reaches into loop body) |
| For-body `?`                                         | same — provenance reaches into loop body |
| Operand-name diagnostic (`?` on i32 says `on 'x'`)   | M2-positive-test passes |
| IR `__try` identity-lowering arm                     | runs only when arity=1 and callee in tuple — no silent fall-through |
| `_lower_type` Result arm                             | guards on `ty.base == "Result" and len(ty.args) == 2` — unknown generics still loud-fail-raise |
| Gate-3 CR-M1 nested try/finally exception safety     | structure verified (pop-raises won't skip restore) |

---

## Verdict

**NOT CLEAN.**

Gate-4 found **2 HIGH silent miscompiles** (G4-F1 + G4-F2) and
2 LOW defensive parity items + 1 MEDIUM guard scope. Both
HIGHs are end-to-end exit-99 verified.

The cascading-defect rhythm continues: gate-3's fix correctly
distinguished inner-LET from inner-ASSIGN, but the disambiguation
is per-name not per-event (G4-F2), and the scope-vehicle
assumption "every arm body is a Block" is false for bare-expr
match arms (G4-F1).

Gate-5 is needed. Both findings have minimal-diff fix sketches
above; a single follow-up commit should close both without
introducing a new architectural surface (just an inline
`_check_expr_in_block_scope` helper + a parallel
`_result_assigns_block_scopes` stack).

---

## Closure-protocol forward note

The Stage 48 closure ledger's pattern observation
("converging on a sound design") holds — each gate has shrunk
the silent-miscompile surface, but the per-name flat-dict
design has now hit its expressiveness ceiling for any
combinator that mixes Assign + Let on the same name within
one block. Stage 49+ runtime tag is the genuine fix; gate-5
should restore the conservative "drop on any mutation"
behaviour rather than try to engineer further mutation
classification on the static side.

Note also: the F5 deferral (gate-2's member-access path)
remains the right call — same defect class, same Stage 49
target.
