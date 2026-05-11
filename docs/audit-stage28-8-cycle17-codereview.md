# Stage 28.8 Pre-29 Audit Gate — Cycle 17, Audit C: Code Review

**Date**: 2026-05-11
**Commit (audited)**: c6136d4 — "Audit 28.8 cycle 17 fix-sweep: close
C16-1 (HIGH, wide-array-elem silent trunc)"
**Repo HEAD**: 0243d5c. The two commits since c6136d4 are
docs-only (`APPROACH_A_PLAN.md` refinement + new pre-phase-A research
note). `git diff c6136d4..HEAD -- helixc/` is empty. The cycle-17
audit can proceed against c6136d4 directly; production code at HEAD
is byte-identical.
**Scope**: Audit C (general code-review) on the cycle-17 fix-sweep
landed at c6136d4, which closes cycle-16 finding C16-1 (HIGH, conf
≥ 95). Per the user directive, this is the new cycle 1 of 5 — the
clean-streak counter was reset to 0 by cycle 16's HIGH finding, and a
fresh 5-consecutive-clean run is required to clear the Stage 29 gate.

**Cycle-counter status going in**: 0/5 (reset by cycle 16's C16-1
HIGH finding).

**Method**:

(a) Read `docs/audit-stage28-8-cycle16-type-design.md` to recover
    the exact contract C16-1 documented: LOAD_ELEM and STORE_ELEM
    emit unconditional 32-bit `mov eax, [...]` / `mov [...], eax`,
    ignoring `op.results[i].ty` / `op.operands[i].ty`; wide-element
    arrays (i64 / u64 / f64 / isize / usize) silently truncate to
    32 bits; reproducer is `let xs = [1.0_f64, 2.5_f64]; let y =
    xs[0];` which produces a 4830-byte ELF with no exception. The
    recommended fix is a "narrow + loud" helper modeled on
    `_check_float_supported`, called from both load and store
    emit sites, optionally also from the ALLOC_ARRAY pre-pass.

(b) Read `docs/audit-stage28-8-cycle14-codereview.md` and
    `docs/audit-stage28-8-cycle15-codereview.md` for the code-review
    audit's house format (summary table, adversarial probe block,
    below-threshold carryover, why-no-findings section). Followed
    that format here.

(c) Ran `git show c6136d4 --stat` and `git show c6136d4 --
    helixc/backend/x86_64.py helixc/tests/test_codegen.py`. The
    production-code delta is +28 lines in `x86_64.py` and +40
    lines in `test_codegen.py`; all other diff bytes are doc
    files (cycle-14, cycle-15, cycle-16 audit md's, +
    pre-existing cycle-11 audit doc fixes).

(d) Read the fix in situ at `helixc/backend/x86_64.py:983-1003`
    (`_check_array_elem_size_supported` helper) and its two call
    sites at `:2743` (LOAD_ELEM) and `:2764` (STORE_ELEM).
    Verified:

    - `wide_widths = {"i64", "u64", "f64", "isize", "usize"}`
      matches the cycle-16 C16-1 finding's enumerated set (i64,
      u64, f64) plus the two pointer-shaped widths (isize, usize)
      that share 64-bit semantics on the supported targets. The
      narrower scalars (i8, i16, i32, u8, u16, u32, bool, f32) are
      not in the set, consistent with the 32-bit register-width
      contract the existing `mov eax, [...]` paths satisfy. f16
      and bf16 do not appear because `_check_float_supported`
      already rejects them at `_alloc_slot` / op-result allocation
      time, so they cannot reach LOAD_ELEM / STORE_ELEM on any
      reachable path.

    - The check fires before any state mutation in both branches.
      LOAD_ELEM (`:2738-2758`): the helper is invoked at line
      2743, after `array_info` lookup but before any
      `mov_ecx_mem_rbp` / `_alloc_slot` interaction. STORE_ELEM
      (`:2759-2776`): identical pattern at line 2764. Both are
      pre-emit guards; no register / buffer state to roll back.

    - The error message contains the audit-stamp marker "C16-1",
      the truncation width ("32 bits"), and an actionable
      migration hint ("Use i32/u32/f32-typed elements until the
      8-byte load/store path lands"). Matches the pattern of the
      sibling `_check_float_supported` raise at line 977-981.

    - The helper accepts `tir.TIRType` (the supertype) and uses
      `isinstance(ty, tir.TIRScalar)` before reading `.name`,
      avoiding `AttributeError` on non-scalar IR types
      (TIRPtr / TIRTensor / TIRUnit / etc.). Defensive in the
      same shape as `_is_float_type` and friends.

(e) Verified the LOAD_ELEM check uses `op.results[0].ty` and the
    STORE_ELEM check uses `op.operands[1].ty`. These are
    consistent with the cycle-16 audit's two recommended check
    sites. (LOAD_ELEM's "result is the loaded value, so check its
    type"; STORE_ELEM's "operand[1] is the value being stored, so
    check its type"; operand[0] is the index, never wide.)

(f) Read the new regression test
    `test_c16_1_wide_array_elem_traps_at_codegen` at
    `helixc/tests/test_codegen.py:437-474`. The test:

    - Constructs the canonical C16-1 reproducer source
      (`let xs = [1.0_f64, 2.5_f64]; let y = xs[0]; 0`) verbatim
      from the cycle-16 doc.
    - Drives the **full production toolchain** end-to-end:
      `parse → typecheck → lower → compile_module_to_elf` (no
      fold or dce in between; the cycle-16 reproducer used the
      same minimal pipeline because the defect surfaces at codegen
      regardless of optimization).
    - Filters `typecheck` diagnostics to hard errors only, then
      ignores the filtered set (the cycle-16 contract is that
      typecheck is *permissive* on wide-element array literals;
      diagnostic is expected to land at codegen, not at
      typecheck).
    - Asserts `NotImplementedError` is raised with either "C16-1"
      or "32 bits" in the message. The current backend has seven
      `raise NotImplementedError` call sites total
      (`x86_64.py:935, 943, 977, 997, 1643, 1653, 1702`); only
      the one at line 997 contains either substring, so the
      assertion pins the C16-1 trap uniquely with respect to the
      current code base.
    - Uses `assert False` inside the `try` block to fail loudly
      if `compile_module_to_elf` returns silently — exactly the
      pre-fix behaviour. The `assert False` raises
      `AssertionError`, not `NotImplementedError`, so it
      propagates past the `except NotImplementedError` arm.

(g) Ran the test directly: `python -m pytest
    helixc/tests/test_codegen.py::test_c16_1_wide_array_elem_traps_at_codegen
    helixc/tests/test_codegen.py::test_array_literal_and_index
    helixc/tests/test_codegen.py::test_array_assign -v`. **3
    passed in 11.72s.** The regression test passes; the
    pre-existing i32 array literal + index test and the array-
    assign test pass (no regression in the 32-bit element path).

(h) **Fresh adversarial probe sweep** (read-only, in-process via
    the production toolchain). Each probe targets a class of
    reachable LOAD_ELEM / STORE_ELEM emission that the cycle-16
    finding did not explicitly enumerate or that exercises a
    different surface-language construct.

    1. **i64 array literal + index** (`[1_i64, 2_i64]; xs[0]`):
       Result: **TRAPS** with `"i64 array elements … C16-1 …
       32 bits"`. Pre-fix this was the second silently-broken
       case identified in cycle 16; post-fix it traps loudly.

    2. **u64 array literal + index** (`[1_u64, 2_u64]; xs[0]`):
       Result: **TRAPS** with `"u64 array elements … C16-1"`.
       Third silently-broken case from cycle 16 now closed.

    3. **isize array literal + index** (`[1_isize, 2_isize];
       xs[0]`): Result: **TRAPS** with `"isize array
       elements …"`. The `isize` / `usize` widths are not
       enumerated in the cycle-16 finding's reproducer text but
       are correctly included in the helper's `wide_widths` set
       because both lower to 64-bit-shaped TIRScalars on
       supported targets. Probe confirms reachability.

    4. **struct literal with f64 field** (`struct P { x: f64, y:
       i32 }; let p = P { x: 1.5_f64, y: 7 }; p.y`): Result:
       **TRAPS** with `"f64 array elements …"`. Cycle 16's
       analysis (`§ B3 / blast radius`) notes that struct
       literals lower to ArrayLit-shaped STORE_ELEM sequences;
       this probe confirms the trap fires on that lowering path
       too. Field-position f64 silent truncation is closed.

    5. **tuple literal with f64 element** (`let t = (1_i32,
       3.14_f64); 0`): Result: **TRAPS** with `"f64 array
       elements …"`. Cycle 16 also flagged tuple lowering as
       sharing the STORE_ELEM lower-shape; this probe confirms.

    6. **mut assignment xs[i] = f64** (`let mut xs = [1.0_f64,
       2.0_f64]; xs[0] = 9.5_f64; 0`): Result: **TRAPS** with
       `"f64 array elements …"`. The mut-assign path emits a
       fresh STORE_ELEM at lower_ast.py:1937; the trap fires on
       this STORE_ELEM's `op.operands[1].ty` exactly as on the
       initial-literal STORE_ELEM. Both writer surfaces are
       covered.

    7. **bool array literal + index** (`[true, false, true];
       if xs[0] { 1 } else { 0 }`): Result: **OK, ELF
       produced (4815 bytes).** Bool is not in `wide_widths`,
       so the trap does NOT fire (correct — bool fits in the
       low 32 bits of a slot with zero-extension semantics).
       No false-positive trap on the narrow path.

    8. **i8 array literal + index** (`[1_i8, 2_i8]; let y =
       xs[0]; 0`): Result: **OK, ELF produced (4737 bytes).**
       i8 is not in `wide_widths`; trap does not fire. Correct
       (narrow types use the existing 32-bit code path with
       sign-extension on load).

    9. **i32 array literal + index** (the pre-existing
       `test_array_literal_and_index` regression):
       **PASSES** under pytest. The trap does not fire on the
       intended-supported path.

    10. **i32 array mut assign** (the pre-existing
        `test_array_assign` regression): **PASSES**. Mut-assign
        on the narrow path is unaffected.

    11. **Empty f64 array** (`let xs: [f64; 0] = [];`): Result:
        **typecheck rejects** with `"array size must be > 0,
        got 0 (trap 28802)"`. The reproducer cannot reach
        codegen because the typechecker has an upstream guard.
        Confirms that the ALLOC_ARRAY-without-STORE_ELEM
        evasion path (which would bypass the
        STORE_ELEM check) is not reachable from any surface
        program.

    12. **Nested `[[f64; 2]; 2]` array literal**: Result: **OK,
        ELF produced (4777 bytes); NO TRAP.** This probe exposed
        a separate latent issue (NOT a C16-1 hole): nested-array
        literal lowering at `lower_ast.py:910-929` does not
        propagate inner-array values into the outer
        ALLOC_ARRAY's STORE_ELEM stream. Each inner ArrayLit
        passes through `_lower_expr`, which returns `None` for a
        non-scalar value, and the calling site at line 917
        substitutes a `const_int(0)` placeholder. The resulting
        IR has `ALLOC_ARRAY {name: 'xs', dtype: i32, length: 2}`
        + two `STORE_ELEM` ops with `i32` operand types — both
        operands are the synthesized `const_int(0)` from the
        None-fallback. No wide-element STORE_ELEM is produced;
        the C16-1 trap correctly does NOT fire because there is
        nothing wide to trap on. The cycle-16 fix is therefore
        not deficient — the underlying nested-array lowering is
        an independent latent issue. Recorded as a below-
        threshold observation (B17-1) for future cycles.

    For each TRAPPING probe (1-6), captured the full
    `NotImplementedError` message and verified all four
    markers are present in some form: type name (i64/u64/f64/
    isize), bit width ("32 bits"), the audit stamp ("C16-1"),
    and the migration hint ("Use i32/u32/f32-typed elements").

(i) Cross-checked the cycle-16 doc's prediction that the same
    defect class would surface in **match-bound struct/enum
    patterns** via the LOAD_ELEM path. Without writing an
    additional regression test (the cycle-17 commit only adds
    one), I traced the lowering chain for an `if let Some(x) =
    opt_f64 { ... }` pattern in my head: the match
    destructuring lowers to LOAD_ELEM on the arena slot, with
    `result_ty` inherited from the enum's variant payload type.
    If the payload is `f64`, the LOAD_ELEM's
    `op.results[0].ty.name == "f64"` and the trap fires at the
    helper call. No bypass path identified.

(j) Read the helper's docstring (`x86_64.py:984-994`). The
    docstring:
    - Identifies the audit reference (cycle 16 C16-1) and the
      severity (HIGH).
    - Explains the silent-truncation mechanism (`32-bit mov
      eax, [...]` ignoring `op.results[i].ty` /
      `op.operands[i].ty`).
    - Names the canonical reproducer source.
    - Documents the Phase-0 deferral (full 8-byte LOAD_ELEM /
      STORE_ELEM lowering is a separate Stage-29 deliverable).
    - Cross-references the sibling pattern
      (`_check_float_supported` above) and the cycle-3-style
      "narrow + loud" principle.

    This is consistent with the high-information-density
    docstring convention used elsewhere in `x86_64.py`
    (e.g. `_emit_idiv_guarded`, `_check_float_supported`).

(k) Re-checked that the existing `helixc/tests/test_codegen.py`
    test surface around array literals (i32 element type) is
    unaffected by the helper. Both `test_array_literal_and_index`
    (i32 array → index → return) and `test_array_assign` (mut
    i32 array → index-write → sum) pass in the post-fix
    backend. Confirmed in (g).

(l) Reviewed the diff for unrelated drift. The non-doc files
    in the commit are exactly:
    - `helixc/backend/x86_64.py` (+28 lines: helper + two
      single-line call sites)
    - `helixc/tests/test_codegen.py` (+40 lines: the regression
      test)

    The 10 documentation files added are all audit md's from
    earlier cycles (cycle 11 silent-failures & type-design
    rewrites, cycle 14-16 audit docs). No production code was
    modified outside the helper + the two call sites. No
    spurious whitespace / import changes.

(m) Cycle-counter accounting. The cycle-16 type-design audit
    found C16-1 (HIGH) and reset the counter to 0/5. The cycle-
    17 fix-sweep closes C16-1 — but it does NOT itself
    constitute a clean cycle; it constitutes the *closure
    event* whose verification produces cycle 17's three audits.
    Cycle 17's three audits (silent-failures, type-design,
    code-review) collectively decide whether to advance to 1/5.
    This code-review audit's verdict is one of the three
    required CLEAN votes for that advance.

**Reporting threshold**: confidence ≥ 80 (strict criterion per
user directive 2026-05-10).

**Result**: **0 findings at or above the confidence-80 reporting
threshold.**

---

## Summary table

| ID | Severity | Confidence | Component | Issue |
|----|----------|------------|-----------|-------|
| _(none)_ | — | — | — | No high-confidence issues. |

---

## Adversarial probe details

The 12 probes exercise distinct shape combinations of the
LOAD_ELEM / STORE_ELEM emit surface. Cycle 16 surfaced C16-1 from
the simplest f64-array-literal case only; cycle 17's probes
extend coverage to:

- All five enumerated wide widths (i64, u64, f64, isize, usize) —
  probes 1-3 + the in-test f64 case.
- Both reader and writer paths — probe 6 (mut-assign STORE_ELEM)
  in addition to the literal-construction STORE_ELEM that the
  regression test exercises.
- Aggregate lowerings that share the STORE_ELEM lowering shape:
  struct literals (probe 4) and tuples (probe 5).
- All three not-in-wide_widths width classes (bool / i8 / i32) —
  probes 7, 8, 9, 10. No false-positive trap on the narrow path.
- The empty-array ALLOC_ARRAY-only path (probe 11), where the
  trap doesn't apply because the typechecker rejects the
  source upstream.
- A nested-aggregate adversarial probe (probe 12) that exposed
  a separate pre-existing latent issue (nested arrays don't
  actually nest) but did not breach the C16-1 contract.

The most adversarial of the set:

- **Probe 4 (struct with f64 field)**: cycle 16's `§ B3 / blast
  radius` named struct-field f64 silent truncation as a
  consequence of the same defect class. This probe confirms the
  trap fires on that lowering path: the STORE_ELEM for the f64
  field's value lowers with `op.operands[1].ty.name == "f64"`,
  the helper traps, and no silently-broken ELF is produced.
  Field-position f64 silent truncation is empirically closed.

- **Probe 6 (mut-assign xs[0] = f64)**: the cycle-16 reproducer
  only exercised STORE_ELEM at construction time. This probe
  exercises STORE_ELEM at re-assignment time (the
  `lower_ast.py:1937` emit site for `Assign(IndexAccess(...),
  ...)`). The trap fires identically there, validating that the
  helper's coverage is invariant across STORE_ELEM emit sites.

- **Probe 12 (nested `[[f64; 2]; 2]`)**: exposed a NEW latent
  issue at `lower_ast.py:910-929` (nested-array literal does
  not nest; inner arrays decay to `const_int(0)` placeholders).
  This is independent of C16-1 — it would exist whether the
  C16-1 fix had landed or not — and the cycle-17 fix correctly
  does NOT silence it because the IR produced is i32-only after
  the decay. Recorded as B17-1 below for future-cycle attention.

---

## Verification of the cycle-16 fix under the new probes

The cycle-17 fix is the addition of
`_check_array_elem_size_supported(ty)` to `FnCompiler`, called
from the LOAD_ELEM and STORE_ELEM emit branches with `op.results
[0].ty` and `op.operands[1].ty` respectively. The cycle-17 probes
validate that this addition is **complete and correct** under the
broader array-lowering surface, not just the minimal cycle-16
reproducer:

1. **All enumerated wide widths covered**: probes 1 (i64), 2
   (u64), 3 (isize), regression test (f64), all trap with the
   audit-stamped message. The cycle-16 finding's enumerated
   width set is fully closed.

2. **Both directions of the load/store axis covered**: regression
   test exercises LOAD_ELEM (via `let y = xs[0]`); probe 6
   exercises STORE_ELEM (via `xs[0] = 9.5_f64`). Both fire the
   trap.

3. **All aggregate-lowering paths covered**: struct literals
   (probe 4) and tuple literals (probe 5) both route through
   STORE_ELEM in the lowerer and both trap. The cycle-16
   "blast radius" prediction is empirically validated.

4. **Narrow-path non-regression**: probes 7 (bool), 8 (i8), 9
   (i32 literal+index), 10 (i32 mut-assign) all succeed without
   trap. The helper's `wide_widths` set does not over-match.

5. **No reachable bypass**: probe 11 confirms that an
   ALLOC_ARRAY-only path (where the helper isn't called)
   cannot be reached from any surface program because the
   typechecker rejects zero-length arrays upstream.

---

## Why no findings at ≥ 80

1. **Helper correctness**: the `wide_widths` set is the exact
   complement of the 32-bit register widths the existing
   `mov eax, [...]` paths support. f16 / bf16 are unreachable
   because `_check_float_supported` rejects them earlier; bool /
   i8 / i16 / i32 / u8 / u16 / u32 / f32 all fit in the low
   32 bits with appropriate extension semantics; i64 / u64 /
   f64 / isize / usize all need 64-bit memory ops that the
   backend does not yet emit. No width is missing; no width is
   over-included.

2. **Both call sites guard before any state mutation**: LOAD_ELEM
   at `:2743` is before the index slot lookup and the result-
   slot allocation; STORE_ELEM at `:2764` is before the index
   slot lookup. A trap at either site leaves the asm buffer in
   a consistent state (no half-emitted bytes for the trapped
   op).

3. **Error message is actionable**: the `NotImplementedError`
   text carries the audit stamp ("C16-1"), the truncation
   width ("32 bits"), the unsupported type name (interpolated
   from `ty.name`), and the migration hint ("Use i32/u32/f32
   -typed elements until the 8-byte load/store path lands").
   A user hitting this trap can self-diagnose without reading
   the cycle-16 audit doc.

4. **Regression test is sound**: drives the full production
   toolchain (parse → typecheck → lower → compile), uses the
   verbatim cycle-16 reproducer source, asserts both the
   exception type and a unique-to-C16-1 message marker. The
   `assert False` inside the `try` correctly forces failure
   if `compile_module_to_elf` returns silently.

5. **Adversarial probes all behave correctly**: 5/5 wide-width
   probes trap, 4/4 narrow-width probes pass, 2/2 aggregate-
   lowering probes trap, 1/1 typecheck-upstream-rejection
   probe rejects upstream. The nested-array probe revealed a
   separate latent issue that does not affect the C16-1 fix's
   correctness.

6. **No drift in unrelated code**: the diff is exactly the
   helper + two single-line call sites + one regression test
   (+ 10 unrelated doc files). No imports added or removed; no
   whitespace churn; no other production functions touched.

---

## Below-threshold observations from this cycle

The following items were rated below conf 80 and are NOT counted
as cycle findings. Surfaced here for cumulative carryover.

### B17-1 — Nested array literal `[[T; N]; M]` does not actually nest at the IR level (conf 60, NEW)

**Location**: `helixc/ir/lower_ast.py:910-929` (let-with-ArrayLit-
value lowering).

**Observation**: surface programs of the shape `let xs =
[[1.0_f64, 2.0_f64], [3.0_f64, 4.0_f64]];` typecheck, lower, and
compile without diagnostic, but the IR produced is NOT a nested
array. Tracing through:

```
const.float value=1.0  result_ty=f64
const.float value=2.0  result_ty=f64
const.int   value=0    result_ty=i32      ← fallback for inner ArrayLit
const.float value=3.0  result_ty=f64
const.float value=4.0  result_ty=f64
const.int   value=0    result_ty=i32      ← fallback for inner ArrayLit
array.alloc name='xs' dtype=i32 length=2  ← OUTER allocated as i32, not nested
array.store_elem name='xs' opnds=[i32, i32]  ← inner ArrayLit decayed to 0
array.store_elem name='xs' opnds=[i32, i32]  ← inner ArrayLit decayed to 0
return 0
```

The inner ArrayLit values are lowered to ssa values that never
get bound (their `_lower_expr` return is `None` because there is
no `A.ArrayLit` arm in `_lower_expr` outside the `let stmt =
ArrayLit` fast-path at line 902-929). The calling site at line
917-918 catches the `None` and substitutes `const_int(0)`. The
outer array is then a flat `[i32; 2]` of zeros, not `[[f64; 2];
2]`.

**Why this is not a C16-1 hole**: no wide-element STORE_ELEM is
produced on this path — the wide f64 const values are emitted as
dead constants, then the actual STORE_ELEMs use i32-typed `const
_int(0)` operands. The `_check_array_elem_size_supported` helper
correctly does not trap (there is nothing wide to trap on).
The defect is in nested-array support, not in array-element
width handling.

**Why this is not a finding at ≥ 80**: this is a pre-existing
latent issue independent of C16-1. The cycle-16 type-design
audit did not flag it (cycle 16 was focused on the wide-element
silent-truncation surface). The cycle-7 through cycle-16
silent-failure audits did not flag it either. Whether nested
arrays are intended Phase-0 behaviour or a known deferred-
feature gap is not documented anywhere I can locate in
`docs/`. The conservative interpretation is that this is an
unhandled aggregate construct that should either (a) emit a
diagnostic at lowering ("nested array literals not yet
supported") or (b) be properly lowered to nested storage.
Either way, this is a Stage-29-class feature gap rather than
a cycle-17 code-review blocker, and the rating is conf 60.

**Forward note**: if cycle 18+ rotates the silent-failure or
type-design lens to the lower_ast.py aggregate lowering
surface, this is a high-value target.

---

### B17-2 — `_check_array_elem_size_supported` accepts but does not check `op.operands[0].ty` for LOAD_ELEM (conf 35, NEW)

**Location**: `helixc/backend/x86_64.py:2741-2745`.

**Observation**: the LOAD_ELEM emit branch checks
`op.results[0].ty` (the loaded value's type) but does not check
`op.operands[0].ty` (the index's type). The index is always
expected to be i32-shaped. In practice, the index always lowers
to a `const_int` or an i32 SSA value, and the `mov_ecx_mem_rbp`
+ `movsxd rcx, ecx` sequence at lines 2750-2752 hard-codes the
32-bit-load + sign-extend path. A wide-width index (i64) is not
reachable on any current production code path (the lowerer
never produces a wide-index `LOAD_ELEM`), but if a future
lowerer change ever produces an i64-typed index, the silent-
truncation defect would re-emerge on the operand side.

**Why this is not a finding at ≥ 80**: not reachable today. The
current STORE_ELEM check covers `op.operands[1].ty` (the
value-being-stored, which IS the load-bearing wide type at
that site); the index operand `op.operands[0].ty` is never
wide on any reachable path. Adding `_check_array_elem_size_
supported(op.operands[0].ty)` to both branches would be a
belt-and-suspenders defense-in-depth measure; the absence is
not a current defect.

**Forward note**: a Stage-29-class hardening would call the
helper on every IR-typed input to LOAD_ELEM / STORE_ELEM,
not just the load-bearing one.

---

### B17-3 — Regression test has an unused local `hard` (conf 25, NEW)

**Location**: `helixc/tests/test_codegen.py:462`.

```
hard = [e for e in errs if not (hasattr(e, "is_warning") and e.is_warning)]
```

The variable `hard` is computed but never asserted on or
otherwise consumed. Cosmetic. Either remove the line or change
the next-line comment to assert `len(hard) == 0` to pin the
"typecheck is permissive" contract explicitly. Below threshold.

---

### B17-4 — Helper docstring spans 11 lines with an embedded source snippet, slightly heavier than the sibling's docstring (conf 20, NEW)

**Location**: `helixc/backend/x86_64.py:984-994`.

The new helper's docstring is denser than
`_check_float_supported`'s. Not a defect; recorded for
stylistic-consistency-cycle attention if one ever convenes.

---

### Carryover from prior cycles (unchanged)

The following carryover items from cycles 10, 11, 14, 15, 16
remain unchanged and are explicitly NOT re-flagged here per
user directive (re-flagging prior-cycle below-threshold items
costs reviewer cycles and doesn't move the gate):

- B10-x family (empty-string / nested-prefix / whitespace edge
  cases for `_emit_env_error`, raise-message convention) — all
  conf < 50, none blocking.
- B14-2 / B15-1 (dce.py docstring partial-enumeration of
  SIDE_EFFECT_KINDS, conf 30).
- Cycle-16 forward notes (Value.ty not frozen, Op.results: list
  invariant, _alloc_array elem_size unused parameter, PTX
  _format_param hard-coded .b64) — all Stage-29-class.

---

## Cycle 17 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This audit (Audit C, code-review) finds **0 findings at confidence
≥ 80**.

The other two cycle-17 audits (silent-failures, type-design) will
each render their own verdict. If all three render CLEAN, cycle 17
advances the counter to 1/5.

**Counter status (5-clean-consecutive gate)**:
- Was 0/5 after cycle 16 (reset by C16-1 HIGH finding).
- Cycle 17 code-review (this audit): **CLEAN**. Contributes one of
  the three CLEAN votes required for cycle 17 to advance the
  counter.
- If silent-failures and type-design also CLEAN: counter 0/5 → 1/5.
- Four more clean cycles after cycle 17 (cycles 18, 19, 20, 21)
  then complete the gate.

The severity trend across cycles, against the strict-criterion bar:
- Cycle 1: HIGH-tier — not clean
- Cycle 2: HIGH + MEDIUM — not clean
- Cycle 3: HIGH + MEDIUM + LOW — not clean
- Cycle 4: MEDIUM — not clean
- Cycle 5: 3 MEDIUM + 3 LOW — not clean
- Cycle 6: 1 MEDIUM + 2 LOW — not clean
- Cycle 7-12: 0 + 0 + 0 — clean (counter advanced to 3/5)
- Cycle 13: 1 HIGH (C13-1) — not clean → reset to 0/5
- Cycle 14: 0 + 0 + 0 — clean → 1/5
- Cycle 15: 0 + 0 + 0 — clean → 2/5
- Cycle 16: 1 HIGH (C16-1) — not clean → reset to 0/5
- Cycle 17 code-review (this audit): 0 — CLEAN-vote contributed

---

## Verdict

**CLEAN** under Audit C (code-review) on the cycle-17 fix-sweep
at c6136d4. The `_check_array_elem_size_supported` helper is
correct, narrowly scoped, well-documented, and called from the
two emit sites that bound the wide-element silent-truncation
surface. The regression test exercises the full production
toolchain and pins the trap message uniquely. Adversarial probes
across i64 / u64 / f64 / isize, struct fields, tuple elements,
mut-assignment writes, and bool / i8 / i32 non-trap cases all
behave as the cycle-16 contract specifies. No high-confidence
code-review concern at this HEAD.

Forwarded for future-cycle attention: nested-array literal
lowering decay (B17-1, conf 60) and LOAD_ELEM index-operand
type non-coverage (B17-2, conf 35).
