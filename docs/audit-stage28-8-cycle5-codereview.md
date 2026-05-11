# Stage 28.8 Pre-29 Audit Gate — Cycle 5, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: 960303b (read-only)
**Scope**: Audit C (general code-review) of the entire compiler stack (stages
1-28.7), with focus on the cycle-4 fix-sweep at commit 960303b that landed 13
fixes (C4-1..C4-5 + E1..E8). Specifically reviewed:

- `helixc/frontend/typecheck.py` — E1 (TyArray size via `_compatible`), C4-4
  (`_compatible` TyTile/TyTensor arms), E2 (Logic-wrap-asymmetric gate
  broadening), C4-1 declarations (TRAP_* constants), E4 (`_resolve_size_expr`
  diag wording).
- `helixc/frontend/monomorphize.py` — C4-5 / E3 (`monomorphize_safe` wrapper +
  `ShapeFoldError.trap_id` class attribute + `TRAP_SHAPE_FOLD_ZERO_DIV`).
- `helixc/frontend/autodiff.py` — C4-1 (Path/Continue/TileLit identity arms),
  C4-3 (If.cond inlining), E6 (Call alias generic-preservation), E7 (TileLit
  identity, covered by C4-1), E8 (Call walks Field-typed callees), the
  catch-all `_ad_warn` cleanup.
- `helixc/bootstrap/parser.hx` — C4-2 (8-arm secondary dispatch widening the
  tag-12 sentinel from val_tag==16 only to Binary/Unary/Index/Field/If/Match/
  Block/UnsafeBlock, with comparison ops marked tag-0).
- `helixc/backend/x86_64.py` — driver switch from `monomorphize` to
  `monomorphize_safe` + diag printing.
- `helixc/bootstrap/kovc.hx` — audit-stage5-6 F9 subpat idx>15 disp8-wrap trap.

**Method**: Read every commit in `b3504a2..960303b` in full. Walked each
modified source file at HEAD. Verified the `A.Expr`/`Type` class shapes
against the new structural arms (typecheck-side `Type` vs ast_nodes-side
`TyNode` distinction matters — they have different `device`/`layout`/`memspace`
field types). Cross-referenced every new `TRAP_*` constant against the
emission sites in source. Verified that `_inline_lets` Call E6 fix handles
the documented reproducer + adjacent cases (turbofish empty list, Path
callee, Field callee). Mentally executed `monomorphize_safe` on a program
where `Monomorphizer.run()` mutates `item.body` for items 0..k-1 before
raising `ShapeFoldError` at item k. Traced the x86_64 driver's post-mono
behavior on the partially-mutated `prog`. Audited the new C4-2 parser.hx
8-arm chain against the AST tag header (lines 11-87 of `parser.hx`) for
val_tag → ty_tag mapping correctness + capture-site `> 0` guard
compatibility. Spot-checked test directory for cycle-4 regression coverage.

**Reporting threshold**: confidence ≥ 80 (per cycle-5 audit-C prompt's
strict criterion).

**Result**: **4 findings (0 CRITICAL, 1 HIGH, 3 MEDIUM, 0 LOW)**.

---

## Summary table

| ID    | Severity | Confidence | Component                                | Issue (short)                                                                                          |
|-------|----------|------------|------------------------------------------|--------------------------------------------------------------------------------------------------------|
| C5-1  | HIGH     | 90         | `helixc/frontend/typecheck.py:221-222`   | Two newly-added `TRAP_*` constants are dead — declared but never referenced; emit sites still use literal `"28802"`/`"28803"` |
| C5-2  | MEDIUM   | 92         | `helixc/backend/x86_64.py:3025-3027`     | `monomorphize_safe` recovery leaves `prog` in a partially-mutated state — driver continues codegen past a fatal trap |
| C5-3  | MEDIUM   | 85         | `helixc/frontend/typecheck.py:1349`      | E2 broadened ONLY the Logic-wrap gate to `l_is_logic != r_is_logic`; the parallel D-wrap-asymmetric same-inner gate (`D<f64> + f64`) is still silent |
| C5-4  | MEDIUM   | 82         | Whole cycle-4 fix-sweep (`960303b`)      | 13 behavioral changes landed with **zero** regression tests; cycle-3 baseline was 16 tests for comparable scope |

**Cycle 5 Audit C: NOT CLEAN — 4 findings (0 CRITICAL, 1 HIGH, 3 MEDIUM, 0 LOW).**

Per user directive 2026-05-10 (strict criterion): cycle counts CLEAN only
when zero findings of ANY severity at or above the audit threshold.

---

## Per-finding sections

### Finding C5-1 — Two new `TRAP_*` constants are dead; audit-time invariant violated

**File**: `helixc/frontend/typecheck.py:221-222` (declared) vs
`helixc/frontend/typecheck.py:591, 596, 606, 611, 2100, 2113` (emit sites).
**Severity**: HIGH
**Confidence**: 90
**Category**: dead code / doc-source mismatch / audit-time-invariant violation

**Description**:
Commit `a59e233` (cycle-4 audit-C C4-1 fix) was specifically intended to
close the prior cycle-4 reviewer's HIGH finding that `docs/lang/trap-ids.md`
rows 28802 / 28803 named `TRAP_*` constants that did not exist in source.
The fix landed module-level declarations:

```python
# helixc/frontend/typecheck.py:221-222
TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO = 28802  # _resolve_size_expr
TRAP_CAST_MATRIX_RECURSION_DEPTH = 28803  # _check_cast_compat
```

but stopped after the declaration half. Repo-wide grep for either
identifier returns exactly one hit (the definition line) and zero
references. Every emit site embeds the trap number as a literal `"trap
28802"` / `"trap 28803"` substring inside an f-string:

- `typecheck.py:591` — `_resolve_size_expr` negative-IntLit branch
- `typecheck.py:596` — `_resolve_size_expr` zero-IntLit branch
- `typecheck.py:606` — `_resolve_size_expr` Unary(-, IntLit) negative branch
- `typecheck.py:611` — `_resolve_size_expr` Unary(-, IntLit) zero branch
- `typecheck.py:2100` — `_check_cast_compat` depth-exceeded structured emit
- `typecheck.py:2113` — `_check_cast_compat` defensive depth-guard

(Note: `TRAP_SHAPE_FOLD_ZERO_DIV` at `monomorphize.py:66` is referenced once
as a class attribute initializer `ShapeFoldError.trap_id`, so it scrapes
past the "at least one caller" half of the invariant — but its emit sites
at lines 117 and 125 also use the literal `"trap 28801"` string, so it
shares the same diagnostic-quality issue at a sub-threshold confidence.)

**Why this matters**:

The registry at `docs/lang/trap-ids.md` declares two audit-time invariants
(lines 94-96). The literal text of the first is:

> Every `TRAP_*` constant must have at least one caller that actually emits
> it. Audit C1 cycle 1 found `@trace` reserved 25001 but never invoked it
> (now fixed in commit c418fb2).

The cycle-1 precedent (`c418fb2`) was specifically called out. The cycle-4
audit doc proposed two fix paths in its Finding C4-1:

> (a) add the three module-level constants and reference them from the
> emission sites, or (b) change the `Constant name` column for these rows
> to `(monomorphize)` / `(typecheck)`.

Cycle 4 took option (a), but only did the declaration half. The
"reference them from the emission sites" half never landed.

Future cycles running the registry's audit-time-invariants sweep will
catch this — exactly as cycle 1 caught `TRAP_TRACE_OVERFLOW`. The fix-
sweep that resolved a HIGH thus regressed into the same class of finding
it was meant to close.

Practical UX harm: mnemonic-driven debugging is dead. A developer who
greps for `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` in source lands on a line
that imports nothing and is imported by nothing. The "Constant name"
column in trap-ids.md becomes a false-pointer at the moment of use.

**Reproducer**:
```sh
$ grep -rn TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO helixc/
helixc/frontend/typecheck.py:221:TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO = 28802  # _resolve_size_expr
$ grep -rn TRAP_CAST_MATRIX_RECURSION_DEPTH helixc/
helixc/frontend/typecheck.py:222:TRAP_CAST_MATRIX_RECURSION_DEPTH = 28803  # _check_cast_compat
```

Single hit each, both pointing only at the declaration line itself.

**Recommended fix** (one of):

- **(a-complete)** Wire the constants into the f-strings: replace each
  `f"(trap 28802)"` with `f"(trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})"`
  (six sites for 28802, two for 28803). Optionally do the same for
  `TRAP_SHAPE_FOLD_ZERO_DIV` at `monomorphize.py:117, 125` for symmetry.
  Matches the precedent set by `TRAP_AD_ASSUMED_ZERO` (referenced from
  `_ad_warn` directly).
- **(b)** Delete the two dead declarations and revert the trap-ids.md
  rows' `Constant name` columns to the parenthesised `(typecheck)`
  convention used by rows 28603 / 28604 / 24001 / 24100 / 24200 / 10030
  / 11001 / 16003.

Option (a) better matches the rest of the registry; option (b) is the
smaller diff.

---

### Finding C5-2 — `monomorphize_safe` recovery leaves `prog` in a partial-mutation state; driver continues past a fatal trap

**File**: `helixc/frontend/monomorphize.py:412-438` (`Monomorphizer.run` body)
  + `helixc/frontend/monomorphize.py:691-706` (`monomorphize_safe` wrapper)
  + `helixc/backend/x86_64.py:3025-3030` (sole caller; warning-and-continue).
**Severity**: MEDIUM
**Confidence**: 92
**Category**: correctness / state mutation through exception / silent
  pipeline continuation on a fatal error

**Description**:
`Monomorphizer.run()` mutates `self.prog` in-place. Two distinct mutation
sites:

```python
# helixc/frontend/monomorphize.py:426-433
while changed:
    changed = False
    for item in list(self.prog.items):
        if isinstance(item, A.FnDecl):
            new_body = self._rewrite_calls_in_block(item.body, item)  # may raise ShapeFoldError
            if new_body is not item.body:
                item.body = new_body          # MUTATION #1: in-place body rewrite
                changed = True
# line 437
self.prog.items = list(self.prog.items) + list(self.instantiated.values())  # MUTATION #2
```

If `_rewrite_calls_in_block` raises `ShapeFoldError` partway through (e.g.
the k-th item triggers a divide-by-zero in a shape constant fold during
generic instantiation), the program state when the exception propagates
out of `Monomorphizer.run()` is:

- Items 0..k-1 of `self.prog.items` may have `item.body` rewritten to
  refer to mangled instantiation names (e.g. `f_T_i32_i32`).
- `self.instantiated` holds the FnDecl clones for those mangled names.
- Line 437 has **NOT** executed — so `self.prog.items` does NOT contain
  the instantiation clones.

The mangled callees referenced by the rewritten bodies thus point at
fns that do not exist in `prog.items`.

`monomorphize_safe` catches `ShapeFoldError` and returns `(0, [str(e)])`:

```python
# helixc/frontend/monomorphize.py:702-706
try:
    return monomorphize(prog), []
except ShapeFoldError as e:
    return 0, [str(e)]
```

The reported count is 0, which is misleading — the underlying program
has been partially mutated. The driver then continues:

```python
# helixc/backend/x86_64.py:3025-3030
mono_count, mono_diags = monomorphize_safe(prog)
for d in mono_diags:
    print(f"warning: fn-mono: {d}", file=sys.stderr)
if mono_count > 0:
    print(f"mono: {mono_count} generic instantiation(s)", file=sys.stderr)
grad_count = grad_pass(prog)
# ... typecheck, hash_cons, totality, lowering, codegen — all on the corrupt prog
```

Two cascading consequences:

1. **Pipeline continues on a partially-mutated prog**. The docstring on
   `monomorphize_safe` (lines 691-702) explicitly says: *"the caller
   should treat the diag as a typecheck error and abort the pipeline
   (callers that don't care can ignore diags; the count is 0 in that
   case)."* The sole caller (`backend/x86_64.py:3026-3027`) ignores the
   docstring and prints the diag as `warning: fn-mono: ...` — same
   severity as struct-mono diags above. Following passes operate on a
   `prog` that has orphaned mangled call-targets.

2. **Subsequent typecheck warnings drown the trap-28801 diagnostic**.
   `grad_pass` / `typecheck` / `lower` see `Call(callee=Name("f_T_i32"))`
   nodes referencing a mangled fn that is not in `prog.items`. The
   user gets the original trap-28801 line as a `warning:`, PLUS a
   typecheck cascade for the orphaned mangled callees, PLUS — if the
   downstream passes happen to typecheck OK after the orphan-warns — a
   produced `out.elf` binary. The user wanted "your `[T; 1/0]` is
   invalid"; they get a noisy warning stream and a binary.

The `monomorphize_safe` design is the cycle-4 *intended* fix for C4-5 /
E3; the cycle-4 doc cited *"x86_64 driver now surfaces the trap-28801
diagnostic cleanly instead of as a compiler-internal-error"* as the
contract. That contract is violated by the driver-side warning-vs-error
choice combined with the in-place mutation that the wrapper does not
roll back.

**Why this matters**:

This is the second cycle in a row where a fix that promises "surface
the diagnostic cleanly" ships without the abort half. Cycle 3 C3-3
added the `_main_inner` try/finally in `check.py` that catches the
*outer* exception and emits "internal error / compiler bug" — the
cycle-4 C4-5 fix-sweep specifically named that misattribution as the
problem-to-solve. The current state is a half-step: the
*misattribution* is closed (the trap-28801 text comes through verbatim),
but the *abort* is not honored at the driver — the user still gets a
binary.

The blast radius is restricted to the developer-debug entry point
(`python -m helixc.backend.x86_64 foo.hx`). The production `helixc
check` CLI (`helixc/check.py:377`) calls only `monomorphize_structs`,
not fn-mono. So this won't bite a normal user — only somebody running
the dev driver on a file that triggers a shape-time div-by-zero
during fn-mono.

The orphaned-mangled-callee state is testable: a one-line program
`fn f[N: usize]() -> [i32; 1/N] { /* body */ } fn main() -> i32 { f::<0>(); 0 }`
would (after E3 lands) produce the trap-28801 diag, then continue
into grad_pass/typecheck/codegen with `prog.items` referring to
mangled callees that aren't present.

**Reproducer** (verbal trace — the regression-test infrastructure for
this path was not added; see C5-4):

1. Write a generic fn whose body produces a Call whose arg flows through
   shape-time fold: e.g. `fn f[N: usize]() -> [i32; 1/N] { ... }` and
   a use site `fn main() -> i32 { f::<0>(); 0 }`. (Or a multi-fn variant
   where item-i has clean shape but item-j has `1/0`.)
2. Run via `python -m helixc.backend.x86_64 repro.hx`.
3. Expected (per E3 contract): `error: fn-mono: ...: division by zero in
   shape expression (trap 28801)` followed by abort + non-zero exit.
4. Actual: `warning: fn-mono: ...: division by zero in shape expression
   (trap 28801)`, then `grad_pass` runs on the partially-mutated `prog`,
   then `typecheck` sees orphan mangled callees, then codegen attempts
   to emit and either crashes or produces a broken `out.elf`.

**Recommended fix**:

Option A (smallest diff, correct semantics): make `monomorphize_safe`
abort by raising or by setting a definitive sentinel that the driver
must respect. Replace the try/except return-tuple with a sentinel
exit-code that the driver `sys.exit(1)`s on, OR keep the tuple but
print `error:` (not `warning:`) and `sys.exit(1)` immediately after
detecting a non-empty diag list.

Option B (smaller scope change but more invasive): make
`Monomorphizer.run()` accumulate the body rewrites into a side-table
and apply them atomically only at the end (transactional semantics).
Then a mid-iteration exception leaves `prog` unmutated.

Option A is the lighter-touch fix and matches the docstring intent.
Option A is also where the structured "abort the pipeline" wording
in the cycle-4 commit message points.

---

### Finding C5-3 — E2 broadened the Logic-wrap gate but not the parallel D-wrap gate; `D<f64> + f64` (same inner, asymmetric D-wrap) is still silent

**File**: `helixc/frontend/typecheck.py:1349` (D-wrap branch gate) vs
  `helixc/frontend/typecheck.py:1363-1366` (Logic-wrap branch gate).
**Severity**: MEDIUM
**Confidence**: 85
**Category**: incomplete-fix / symmetry violation between Logic-domain and
  D-domain mixing gates

**Description**:
Cycle 4 E2 broadened the Logic-wrap-asymmetric warning gate. Before E2:

```python
elif (l_is_logic or r_is_logic) and inner_mismatch:
    # warn — Logic + raw with different inner
```

After E2:

```python
# helixc/frontend/typecheck.py:1363-1366
elif (l_is_logic or r_is_logic) and (
        inner_mismatch
        or (l_is_logic != r_is_logic)
):
```

i.e. fire when either inner differs OR the wrap is asymmetric. This closes
the `Logic<f64> + f64` (same inner, asymmetric Logic-wrap) silent-acceptance
case.

The parallel D-wrap branch (line 1349) was **not** broadened:

```python
# helixc/frontend/typecheck.py:1349
if (l_is_diff or r_is_diff) and inner_mismatch:
    # warn
```

So `D<f64> + f64` (same inner, asymmetric D-wrap) falls through to the
`else` clause at line 1391-1393 and silently picks the inner without a
warning:

```python
else:
    inner = l_inner if not isinstance(l_inner, TyUnknown) \
        else r_inner
```

The cycle-4 commit message for E2 calls this out explicitly:

> E2: Logic+bare-T wrap-asymmetric now warns (gate broadened to fire when
> l_is_logic != r_is_logic, regardless of inner). Symmetric with cycle-2
> B:C6 D-vs-bare pattern.

The "symmetric with cycle-2 B:C6 D-vs-bare pattern" claim is exactly wrong:
cycle-2 B:C6 closed the *inner-mismatch* asymmetric case (`D<f64> + i32`),
but the same-inner asymmetric case (`D<f64> + f64`) was never closed for
D-wrap, and E2 only ported the Logic-wrap fix. The mirror gap remains.

**Why this matters**:

`TyDiff` is the autodiff wrap; raw-vs-wrapped at the same inner type means
the user wrote a binop between a differentiable value and a non-differentiable
constant of matching dtype. This is the same documented foot-gun as
`Logic<f64> + f64` (E2's intended target): silently dropping the wrap
loses provenance for the downstream `grad()` call.

The cycle-2 B:C6 fix message says "asymmetric D<T> + bareT also warns" —
which is true for **mismatched inner** but not for **same inner**. A
reader who trusts cycle-2 B:C6's promise will assume the case is closed,
and the cycle-4 E2 commit message reinforces that assumption (claiming
parallel symmetry).

**Reproducer**:
```helix
fn loss(x: D<f64>) -> D<f64> {
    let c: f64 = 1.0;
    return x + c;          // D<f64> + f64, same inner, no warning
}
```

Pre-cycle-4: silent. Post-cycle-4 (post-E2): still silent for D-wrap;
the analogous `Logic<f64> + f64` would now warn.

**Recommended fix**:

Mirror the E2 broadening at the D-wrap gate. Replace line 1349 with:

```python
if (l_is_diff or r_is_diff) and (
        inner_mismatch
        or (l_is_diff != r_is_diff)
):
```

The downstream `extra = " (one side D-wrapped, other bare)"` annotation
at line 1359 already handles the asymmetric case; the gate is the only
piece missing.

Also worth: add the parallel `if not (l_is_diff and r_is_diff)` clause
to the asymmetric-with-same-inner case to suppress the inner-widen warning
(which would re-fire pointlessly on same-inner). The Logic-branch already
does this at line 1385.

---

### Finding C5-4 — Cycle-4 fix-sweep landed 13 behavioral fixes with zero regression tests

**File**: Cycle-4 commit `960303b` overall (+ ancestry `b3504a2..960303b`).
**Severity**: MEDIUM
**Confidence**: 82
**Category**: test-coverage / regression-baseline drift

**Description**:
The cycle-4 fix-sweep documents 13 distinct findings closed (HIGH C4-1,
C4-2; MEDIUM C4-3, C4-4, E1, E2, E3/C4-5; LOW C4-1-sweep'd E7, E4, E6,
E8). Each is a real behavioral change:

- **C4-1**: `_inline_lets` adds three identity arms (`A.Path`, `A.Continue`,
  `A.TileLit`) before a hot catch-all in the AD pass.
- **C4-2**: `parser.hx` lines 2334-2374 add a 41-line, 8-arm secondary
  dispatch table classifying eight new val_tag families (Binary,
  Unary, Index, Field, If, Match, Block, UnsafeBlock) as untracked-complex
  sentinel (tag 12) or proven-i32 bools (AST_LT/GT/EQ/NE/LE/GE → tag 0).
  Each new arm is a new closure-capture trap-trigger trajectory.
- **C4-3**: `_inline_lets` now recurses through `A.If.cond` (previously
  passed through unmodified).
- **C4-4**: `_compatible` adds two structural arms (`TyTensor` and
  `TyTile`) with shape/dtype/device/layout/memspace comparison.
- **E1**: `_compatible.TyArray` upgrades raw `a.size == b.size` to a
  recursive `_compatible(a.size, b.size)` fallback.
- **E2**: Logic-binop tie-callback gate broadened from `inner_mismatch`
  only to `inner_mismatch or (l_is_logic != r_is_logic)`.
- **E3 / C4-5**: New public function `monomorphize_safe` wraps
  `monomorphize` and catches `ShapeFoldError` into a diags list. New
  module-level constant `TRAP_SHAPE_FOLD_ZERO_DIV` (= 28801) +
  `ShapeFoldError.trap_id` class attribute.
- **E4**: `_resolve_size_expr` diag wording unified to "must be > 0" in
  both branches (cosmetic).
- **E6**: `_inline_lets.A.Call` arm preserves caller turbofish generics
  on alias substitution.
- **E8**: `_inline_lets.A.Call` arm walks `A.Field`-typed callees.

Grep of `helixc/tests/` for any of `C4-1`, `C4-2`, `C4-3`, `C4-4`, `C4-5`,
`E1`, `E2`, `E3`, `E4`, `E6`, `E7`, `E8`, `monomorphize_safe`,
`TRAP_SHAPE_FOLD_ZERO_DIV`, `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO`,
`TRAP_CAST_MATRIX_RECURSION_DEPTH` returns zero matches. The only test
edits in the b3504a2..960303b range are:

- `db4055b` (audit-stage5-6 F9 codegen test) — retroactive close of a
  *prior* stage-5-6 audit finding (the disp8 wrap), not a cycle-4
  regression test.
- `31c7912` const-fold dedup — removes duplicate tests, doesn't add.

The cycle-4 commit message claims *"Tests: targeted suite (autodiff +
typecheck + struct_mono) 172 pass."* — these are existing tests.

The cycle-3 fix-sweep (compare commit ranges) added 16 regression tests
for a comparable set of findings; the cycle-4 audit-C cycle-4 doc
explicitly noted (axis 3) *"15 new regression tests for cycle-3 findings
+ the C3-1 test committed earlier… Spot-check of 5 …shows both happy
and error paths covered. PASS."*

**Why this matters**:

1. **Regression baseline drift**. The cycle-4 fix-sweep is the documented
   sole vehicle for closing audit findings during this gate cycle. Cycle
   3 set the bar at one-test-per-finding (occasionally more). Cycle 4
   lands at zero. A future audit cycle that retroactively flags a
   regression in any of these 13 fixes will have no test to bisect
   against; cycle-3 fixes are bisectable by name (`test_c3_3_*`,
   `test_d6_*`), cycle-4 fixes are co-mingled in the existing test-suite
   happy-path coverage at best.

2. **Bootstrap-language risk concentration**. C4-2 is the highest-risk
   modification — bootstrap-language changes have historically been the
   highest-risk class (per audit-stage5-6 F9, audit-stage7-8 F4/F12, the
   cycle-3 D2 fix itself). The new 8-arm dispatch covers eight val_tag
   families. None has a dedicated regression test asserting the val_tag
   → inferred_ty_tag mapping or the downstream trap-76003 firing. The
   cycle-3 D2 fix (which C4-2 extends) had a dedicated test (`test_codegen
   .py` D2 case asserting exit 132 on `let pi = get_pi(); let c = |y| y
   + pi; c(0)`); the natural cycle-4 analogues are absent.

3. **`monomorphize_safe` semantic drift uncovered** (interacts with C5-2
   above). The new wrapper's docstring contract is not enforced by any
   test; the cycle-4 fix-sweep doesn't add a happy-path test asserting
   the (count, diags) shape on a clean run, nor a fail-path asserting
   the trap-28801 message and the count==0 zero-instantiation case. A
   test like `def test_e3_monomorphize_safe_catches_shapefolderror():
   assert monomorphize_safe(prog_with_div0) == (0, [diag_containing_28801])`
   would have caught the C5-2 driver-side severity-downgrade in CI.

**Reproducer**:
```sh
$ grep -rn 'C4-\|monomorphize_safe\|TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO' helixc/tests/
# (empty — except for the helixc-Python-CMP test in test_typecheck which is unrelated)
```

**Recommended fix**:

Add per-finding regression tests in the cycle-5 fix-sweep. Minimal
suggested coverage (one happy-path test per behavioral change):

- `test_c4_1_inline_lets_no_warning_on_path_continue_tilelit`: assert
  `_inline_lets` returns the input unchanged for each of `A.Path`,
  `A.Continue`, `A.TileLit`, with no `_ad_warn` fire.
- `test_c4_2_*` (eight cases, one per new val_tag family): `let x =
  <expr>; let c = |y| y + x; c(0)` where `<expr>` is Binary
  (`1+2`), Unary (`-1`), Index (`arr[0]`), Field (`obj.f`), If-expr,
  Match-expr, Block-expr, UnsafeBlock-expr. Asserts exit 132 = trap
  76003 from the capture probe.
- `test_c4_2_bool_capture_safe`: `let x = a < b; let c = |y| y + x;
  c(0)` — assert clean compile (AST_LT registered as i32-tag).
- `test_c4_3_inline_lets_if_cond`: assert `let g = grad(loss); if
  g(x) > 0.0 { ... }` has `g` substituted in the cond.
- `test_c4_4_compatible_tytensor_dtype_mismatch`,
  `test_c4_4_compatible_tytile_memspace_mismatch`,
  `test_c4_4_compatible_tytensor_shape_mismatch`: structural mismatch
  cases.
- `test_e1_compatible_tyarray_size_via_compatible`: TyArray with
  TyUnknown size on one side does not false-positive.
- `test_e2_logic_wrap_asymmetric_warns`: `Logic<f64> + f64` (same
  inner, asymmetric wrap) now warns where pre-fix it was silent.
- `test_e3_monomorphize_safe_catches_shapefolderror`: program with
  `fn f[N: usize]() -> [i32; 1/N] { ... }` instantiated `f::<0>` —
  assert `(0, [diag])` where diag contains `"28801"`. Also a
  separate test asserting the **driver** behavior (currently passes
  with the wrong semantics — see C5-2).
- `test_e6_inline_lets_call_preserves_generics`: `let g = mk_grad;
  g::<f64>(x)` retains `::<f64>` after alias substitution.
- `test_e8_inline_lets_call_field_callee`: `let obj = make();
  obj.method()` substitutes `obj` correctly.

11-13 tests total — comparable to the cycle-3 baseline.

---

## Files reviewed

`helixc/backend/x86_64.py`, `helixc/bootstrap/kovc.hx`,
`helixc/bootstrap/parser.hx`, `helixc/frontend/ast_nodes.py`,
`helixc/frontend/autodiff.py`, `helixc/frontend/monomorphize.py`,
`helixc/frontend/typecheck.py`, `helixc/tests/test_autodiff.py`,
`helixc/tests/test_codegen.py`, `helixc/tests/test_const_fold.py`,
`helixc/tests/test_struct_mono.py`, `helixc/tests/test_typecheck.py`,
`docs/lang/trap-ids.md`, plus the persisted cycle-4 audit-doc files
(`audit-stage28-8-cycle4-{codereview,silent-failures,type-design}.md`)
and the cycle-5 type-design audit (`audit-stage28-8-cycle5-type-design.md`)
for cross-reference.

---

## Specific cycle-4 changes audited (25 items)

1. `helixc/frontend/monomorphize.py:60-77` — `TRAP_SHAPE_FOLD_ZERO_DIV`
   constant + `ShapeFoldError.trap_id` class attribute. **PARTIAL FAIL**
   (sub-threshold) — constant is technically referenced once (line 76)
   but emit sites at 117, 125 still use literal `"trap 28801"`.
2. `helixc/frontend/monomorphize.py:691-706` — `monomorphize_safe`
   wrapper. **FAIL** (C5-2 — state-leak on caught exception).
3. `helixc/frontend/typecheck.py:217-222` — Two new module-level
   `TRAP_*` constants. **FAIL** (C5-1 — dead).
4. `helixc/frontend/typecheck.py:578-613` — `_resolve_size_expr` E4 diag
   wording unification (both branches now say "must be > 0"). **PASS**.
5. `helixc/frontend/typecheck.py:1349` — D-wrap branch gate (unchanged
   by cycle 4). **FAIL** (C5-3 — should have been mirrored to match E2).
6. `helixc/frontend/typecheck.py:1363-1366` — E2 Logic-wrap gate
   broadening to include `l_is_logic != r_is_logic`. **PASS** for the
   change itself; the asymmetric annotation at 1385-1389 correctly
   suppresses the inner-widen warning for same-inner cases.
7. `helixc/frontend/typecheck.py:2197-2204` — E1 TyArray size via
   `_compatible` fallback. **PASS** for the implementation; structural
   gap (TySize-vs-TySize cascade) is the type-design F1 finding, not a
   code-review issue.
8. `helixc/frontend/typecheck.py:2225-2248` — C4-4 TyTensor / TyTile
   structural arms. **PASS** — `TyTensor.device/layout` and
   `TyTile.memspace` in the typecheck-side `Type` lattice are
   `Optional[str]` / `str` (not `Expr`), so raw `==` is correct. (Note:
   the ast_nodes-side `TyTensor` uses `Optional[Expr]`, which would
   make raw `==` span-sensitive — but the `_compatible` arm operates
   on the resolved `Type` form, not the AST node form. Verified by
   reading both class definitions.)
9. `helixc/frontend/autodiff.py:529-552` — C4-3 If.cond inlining + the
   `new_cond` wiring. **PASS**.
10. `helixc/frontend/autodiff.py:564-590` — E6 Call alias
    generic-preservation + E8 Field-typed callee walk. **PASS** for
    the documented cases. Edge case: when `cand` is `A.Path`, the
    `expr.callee.generics` is dropped (A.Path has no generics field) —
    flagged in the cycle-5 type-design notes at confidence 50, below
    threshold. Recursion termination on Field-callee walk: bounded by
    source nesting depth (e.g. `obj.f.g.h.method()` has depth = chain
    length), safe.
11. `helixc/frontend/autodiff.py:699-723` — C4-1 identity arms for
    Path / Continue / TileLit before the catch-all + catch-all reason
    cleanup (drop the literal `(trap 85001)` substring). **PASS** for
    the documented C4-1/C4-3/C4-5 closure. TileLit identity arm doesn't
    recurse into shape/memspace — F4 in cycle-5 type-design at LOW; not
    re-flagged here (overlap with type-design audit B).
12. `helixc/bootstrap/parser.hx:2334-2374` — C4-2 8-arm secondary
    dispatch widening the tag-12 sentinel. **PASS** for brace balance
    (verified by per-pattern counter); val_tag → inferred_ty_tag
    mapping correctness against the AST tag header (lines 11-87) is
    consistent: 6 (AST_LT) / 19 (AST_GT) / 20-23 (AST_EQ-AST_GE) all
    correctly registered as tag 0 (i32 reified bool); AST_VAR (val_tag
    == 1) correctly left at inferred_ty_tag = -1 to defer to
    var_type_tab_lookup. Capture-site guard at line 1820 (`if cap_ty_tag
    > 0`) correctly treats tag 0 as safe and any positive tag (including
    sentinel 12) as trap-trigger.
13. `helixc/bootstrap/parser.hx:2380-2384` — `var_type_tab_add` only
    fires when `inferred_ty_tag >= 0`. **PASS** (AST_VAR untracked path
    correctly bypasses registration).
14. `helixc/backend/x86_64.py:2979-2982` — driver import switch from
    `monomorphize` to `monomorphize_safe`. **PASS** for the import; the
    use-site at 3025-3027 is the C5-2 finding.
15. `helixc/backend/x86_64.py:3021-3027` — driver diag-print + continue.
    **FAIL** (C5-2 — should abort, not warn).
16. `helixc/bootstrap/kovc.hx:3340-3392` — emit_variant_subpats /
    emit_tuple_subpats audit-stage5-6 F9 disp8-cap trap. **PASS** —
    guard `if idx_in_payload > 15` fires at idx=16 where off becomes
    128 = signed disp8 -128. Trap emitted before the wrapping load.
17. `docs/lang/trap-ids.md` row 28801 — references
    `TRAP_SHAPE_FOLD_ZERO_DIV`. Constant exists at
    `monomorphize.py:66`. **PASS** for the row; the emit-site literal
    is a sub-threshold concern noted in item 1 above.
18. `docs/lang/trap-ids.md` row 28802 — references
    `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO`. **FAIL** via C5-1 (dead).
19. `docs/lang/trap-ids.md` row 28803 — references
    `TRAP_CAST_MATRIX_RECURSION_DEPTH`. **FAIL** via C5-1 (dead).
20. `docs/lang/trap-ids.md` row 76003 — extended for the cycle-3 D2
    Call-RHS-let trigger. **PASS** (closes the prior cycle-4 C4-4 LOW
    finding).
21. `docs/lang/trap-ids.md` header "Last updated" bumped to 2026-05-11.
    **PASS** (closes the prior cycle-4 C4-2 MEDIUM finding).
22. `helixc/frontend/autodiff.py:719-723` — catch-all warning reason
    no longer pre-embeds `(trap 85001)`. **PASS** (closes the prior
    cycle-4 C4-3 MEDIUM finding).
23. `helixc/tests/test_codegen.py:3308-3336` — new F9 regression test
    for subpat idx>15 disp8 wrap. **PASS** (a regression test landed
    for the audit-stage5-6 F9 fix specifically; not a cycle-4 fix
    regression).
24. `helixc/tests/test_const_fold.py` — removal of two duplicate tests
    (`test_x_mod_one_folds_to_zero`, `test_x_div_one_folds`). Verified
    the surviving copies at lines 245-255 cover the same surface area.
    **PASS**.
25. Cycle-4 fix-sweep test count for the 13 new fixes: **0 added.**
    **FAIL** (C5-4 — zero regression tests for cycle-4 fixes).

---

## What was checked and found below threshold

- **`_inline_lets.A.Call` E6 when `cand` is `A.Path`** (line 584-585):
  `new_callee = cand` drops the callsite's `expr.callee.generics`
  because A.Path has no `generics` field. The existing behavior for
  this sub-case matches the pre-fix code. A future Path-with-generics
  extension would resurface this. **Confidence 55**, below threshold.
- **`_inline_lets.A.Call` when `cand` is neither Name nor Path** (line
  588 elif-branch covers Field, but other cand shapes — Lambda, Block,
  Call — leave `new_callee = expr.callee` unchanged). This matches
  the pre-fix design (alias-only substitution); confirmed by the
  inline comment at line 567-569. **PASS / by-design.**
- **`Monomorphizer.run` while-loop termination guarantee** (line
  425-433): the fixed-point invariant is `changed = True` only when
  some `item.body is not item.body` rewrite happened, and each
  rewrite produces a body with strictly fewer call-to-generic-fn
  sites that have unique-type-arg-tuples-not-yet-instantiated. So
  the loop terminates. Pre-cycle-4 invariant; not changed by the
  fix-sweep. **PASS.**
- **`_inline_lets.A.Block.stmts` linear-search `.index(stmt)` (line
  505)**: O(n²) in stmt-count for `_is_reassigned_after`
  start-index lookup. Pre-cycle-4 (introduced by `e25aea9` deep-
  research cycle 4 long before audit 28.8), not modified by cycle 4.
  **Confidence 60**, below threshold; out of cycle-4 scope.
- **`emit_byte(off_in_payload)` in kovc.hx when `off > 255`**: the
  emit accepts an i32 value and presumably masks to a byte. The
  trap fires first at idx > 15, so the wrapping byte is dead code.
  **PASS.**
- **`ShapeFoldError` is a subclass of `ValueError`**: any caller of
  `monomorphize` that catches `ValueError` (rather than
  `ShapeFoldError`) would also catch the typed shape-fold error and
  conflate it with other ValueError sources. Grep of source shows
  only `monomorphize_safe` catches; no other callers do a broad
  `ValueError` catch on the mono path. **PASS / no current conflict.**

---

## Open prior findings (not re-flagged this cycle)

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16 baselines
unchanged from cycle-1 status (see `audit-stage28-8-cycle1-codereview.md`
lines 102-106). Cycle-1 / cycle-2 / cycle-3 / cycle-4 findings: all
marked CLOSED by their respective fix-sweep commits, with these
exceptions reopened this cycle:

- C4-1's option-(a) fix is incomplete → reopened as **C5-1**.

The cycle-5 type-design audit (audit B, finished earlier this
session, file `audit-stage28-8-cycle5-type-design.md`) found 6
findings (0 HIGH, 3 MEDIUM, 3 LOW). C5-3 above overlaps with that
audit's F2 (D-wrap-asymmetric mirror) at the code-review angle
(missing gate broadening), but is flagged here independently because
this is a behavioral correctness issue identifiable from the diff
alone, not just a type-system soundness analysis. C5-2 partially
overlaps with type-design F3 (`monomorphize_safe` severity) but
sharpens the concern from "warning-not-error" to the more concrete
"partial in-place mutation leaks orphaned mangled callees into
subsequent passes" — testable independently of the abort-vs-warning
choice.

The type-design F1 (TyArray-size cascade gap), F4 (TileLit identity
arm not walking shape/memspace), F5 (parser tag-12 namespace overlap),
F6 (C4-2 AST_VAR -1 fallback) are all type-design-soundness or
edge-case-diagnostic-quality issues that are below the code-review
threshold here. Tracked in the type-design audit doc; not re-flagged.

---

## Verdict

**Cycle 5 Audit C: NOT CLEAN — 4 findings (0 CRITICAL, 1 HIGH, 3 MEDIUM, 0 LOW).**

Strict-zero rule per user directive 2026-05-10. Cycle counter does not
advance.

Suggested fix order (smallest blast radius first):

1. **C5-3** (MEDIUM, smallest diff): broaden the D-wrap-asymmetric gate
   at `helixc/frontend/typecheck.py:1349` to mirror the E2 Logic-wrap
   broadening at line 1363-1366. Add the parallel
   `if not (l_is_diff and r_is_diff)` clause for the same-inner
   asymmetric case. ~6-line change.
2. **C5-1** (HIGH, small diff): wire `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO`
   (six emit sites) and `TRAP_CAST_MATRIX_RECURSION_DEPTH` (two emit
   sites) into the f-strings via the `f"(trap {TRAP_*})"` pattern, OR
   delete the dead declarations and revert the trap-ids.md columns to
   `(typecheck)`. Optionally do the same for `TRAP_SHAPE_FOLD_ZERO_DIV`
   at `monomorphize.py:117, 125` for symmetry.
3. **C5-2** (MEDIUM, scope decision required): make the x86_64 driver
   abort on `mono_diags` non-empty (option A — single line:
   `if mono_diags: print(...); sys.exit(1)`) OR make
   `Monomorphizer.run()` transactional (option B — larger refactor).
   Option A is the lighter touch and matches the docstring intent.
4. **C5-4** (MEDIUM, larger): add the 11-13 regression tests listed
   under the C5-4 fix section to `helixc/tests/{test_autodiff.py,
   test_typecheck.py, test_codegen.py, test_monomorphize.py}`. Brings
   cycle-4 test-density to the cycle-3 baseline.

After fixes land, rerun cycles A (silent-failure) + B (type-design) +
C (this audit) for cycle 6.
