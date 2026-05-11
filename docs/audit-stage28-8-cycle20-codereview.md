# Stage 28.8 Pre-29 Audit Gate — Cycle 20, Audit C: Code Review

**Date**: 2026-05-11
**Commit (audited)**: `5a1e406` — "Audit 28.8 cycle 20 fix-sweep:
close C19-1 (HIGH, const_fold isize 32-bit drift)".
**Repo HEAD**: `5a1e406` (no commits since the fix-sweep at audit
time).
**Scope**: Audit C (general code-review) on the cycle-20 fix-sweep
that closes cycle-19 finding C19-1 (HIGH, conf ≥ 95). Per the user
directive, this is the new cycle 1 of 5 — the clean-streak counter
was reset to 0 by cycle 19's HIGH finding (the cycle-19 type-design
audit). A fresh 5-consecutive-clean run is required to clear the
Stage 29 gate.

**Cycle-counter status going in**: 0/5 (reset by cycle 19's C19-1
HIGH finding).

**Method**:

(a) Read `docs/audit-stage28-8-cycle19-type-design.md` to recover
    the exact contract C19-1 documented. The summary:

    - Cycle 19's backend fix at `x86_64.py:1005-1017` made
      `_is_i64_type` / `_is_u64_type` recognize `isize` /
      `usize` as 64-bit (canonicalizing them to i64 / u64 to match
      `typecheck.py:225-228`'s `_WIDEN_NAME_ALIASES`).
    - But `helixc/ir/passes/const_fold.py:46` still carried
      `"isize": 32, "usize": 32` in `_INT_BITS`. That table
      drives `_wrap_int_to_type`, which is the const-fold
      result-wrap function called from every const-folded
      integer arithmetic / bitwise / neg / bit-not op
      (sites at lines 327, 403, 444, 475 pre-fix; lines 334,
      410, 451, 482 post-fix).
    - End-to-end consequence: `let a: isize = 3e9; let b: isize =
      3e9; a + b` (un-folded) returns 6_000_000_000 (correct,
      64-bit ADD); the *same* program with const_fold enabled
      (default `-O1` per `check.py:581`) wraps the folded result
      to 32 bits, producing `1_705_032_704_isize`, which the
      cycle-19 backend then emits as a 64-bit literal. The folded
      path silently miscompiles. **Optimization-unstable**: -O0
      and -O1 disagree.
    - Cycle 19 rated this HIGH (conf ≥ 95). Same defect class as
      C13-1, C16-1, C18-1: one pass's width contract disagrees
      with another's.

(b) Read `docs/audit-stage28-8-cycle18-codereview.md` and
    `docs/audit-stage28-8-cycle17-codereview.md` for the
    code-review audit's house format (summary table, adversarial
    probe block, below-threshold carryover, why-no-findings
    section). Followed that format here.

(c) Ran `git show 5a1e406 --stat` and `git show 5a1e406 --
    helixc/ir/passes/const_fold.py helixc/tests/test_const_fold.py`.
    The production-code delta is **+8 lines** in `const_fold.py`
    (1 line of width edit + 7 lines of audit-stamped comment)
    and **+31 lines** in `test_const_fold.py` (one new regression
    test function `test_c19_1_isize_usize_are_64_bit_in_wrap`).
    The 653-line `docs/audit-stage28-8-cycle19-type-design.md` is
    the cycle-19 type-design audit doc being persisted alongside
    the fix (not part of the production-code surface).

(d) Read the fix in situ at `helixc/ir/passes/const_fold.py:43-56`:

    ```python
    _INT_BITS = {
        "i8": 8, "u8": 8,
        "i16": 16, "u16": 16,
        "i32": 32, "u32": 32,
        # Audit 28.8 cycle 20 C19-1 (HIGH): pointer-width aliases must be
        # 64-bit, matching typecheck.py:225-228's `_widen_canon_name`
        # aliasing (isize->i64, usize->u64) and the cycle-19 backend
        # classifier fix at x86_64.py:1005-1017. Pre-fix the 32-bit
        # entry made `_wrap_int_to_type(6_000_000_000, isize) =
        # 1_705_032_704` — silent miscompile reachable at default -O1.
        "isize": 64, "usize": 64,
        "i64": 64, "u64": 64,
        "bool": 32,  # bool comparisons reified to i32 in IR
    }
    ```

    Verified:

    - The width edit changes `isize` / `usize` from 32 to 64
      and matches the established canon: `typecheck.py:225-228`
      aliases isize→i64 and usize→u64; `typecheck.py:241` ranks
      both at the i64 / u64 widening level; `typecheck.py:1816
      -1817` bounds them at the i64 / u64 numeric range. The
      cycle-19 fix at `x86_64.py:1011, 1017` extends
      `_is_i64_type` / `_is_u64_type` to recognize them. The
      cycle-20 fix completes the canon at the const-fold layer.
    - The audit-stamp comment cites C19-1 (HIGH), names both
      cross-reference sites (typecheck.py line range + cycle-19
      backend fix), and includes the canonical reproducer
      (`_wrap_int_to_type(6_000_000_000, isize) =
      1_705_032_704`) so future readers can grep for the
      defect lineage. Matches the high-information-density
      audit-stamp convention used at the cycle-17 / cycle-19
      sites (`x86_64.py:983-1003`, `x86_64.py:1005-1017`).
    - `bits` default of 32 in `_wrap_int_to_type` is unchanged
      (line 66). This is the fallback for unknown / generic
      scalar types only; no isize/usize codepath reaches it
      post-fix because `_INT_BITS["isize"]` and
      `_INT_BITS["usize"]` are now both present at 64. The
      32-bit fallback is appropriate as a conservative default
      — it matches Phase 0's narrow-int register-width
      contract.

(e) Verified placement: the edit is the **only** production-code
    change in the commit. The `_wrap_int_to_type` body (lines
    59-74) is unchanged — same mask/sign-extend math as cycle 17.
    The function correctly reads `bits = _INT_BITS.get(ty.name,
    32)` at line 68 and proceeds with `mask = (1 << bits) - 1`,
    `half = 1 << (bits - 1)` and two's-complement
    sign-correction. No other adjustment is required.

(f) **Trace every call site of `_wrap_int_to_type`** to confirm
    the fix's reach. Grep at `const_fold.py`:

    - Line 334 (ADD / SUB / MUL / DIV / MOD result wrap): `v =
      _wrap_int_to_type(v, res.ty)`. `res.ty` is the result-Value's
      type, which mirrors the source-language type (the typechecker
      tags the ADD's result with the wider of l-ty and r-ty per
      `typecheck.py:_widen_diff_inner`). For isize/usize ops, `res.ty`
      is `TIRScalar("isize")` / `TIRScalar("usize")`. Post-fix, the
      wrap uses 64 bits.
    - Line 410 (BIT_AND / BIT_OR / BIT_XOR / SHL / SHR result wrap):
      same `res.ty` parameter. Post-fix, isize/usize bitwise ops
      preserve their 64-bit semantics.
    - Line 451 (unary NEG on integer): same. `-2**31_isize` no
      longer wraps to a 32-bit-signed value; it correctly
      preserves the 64-bit two's-complement.
    - Line 482 (BIT_NOT on integer): same. `~0_isize` is now
      `-1_isize` (full 64 bits), not `0xFFFFFFFF` zero-extended.

    All four call sites correctly route through the fixed
    `_INT_BITS` table; the fix has uniform reach across the
    const-fold result-wrap surface.

(g) Read the new regression test
    `test_c19_1_isize_usize_are_64_bit_in_wrap` at
    `helixc/tests/test_const_fold.py:356-384`. The test:

    - Imports `_wrap_int_to_type` directly (white-box on the unit
      under fix), not the higher-level `fold_module`. Appropriate
      for a one-line `_INT_BITS` edit: the unit-level test pins
      the contract at the function whose behavior changed.
    - Pins the canonical reproducer: `_wrap_int_to_type
      (6_000_000_000, isize) == 6_000_000_000` and `_wrap_int_to_type
      (6_000_000_000, i64) == 6_000_000_000`. Pre-fix the isize
      assertion would fail (yields 1_705_032_704); post-fix both
      pass. This is the **proof of fix**.
    - Exercises both classes — isize/i64 agreement and usize/u64
      agreement — across nine and six representative values
      respectively. Values include sign-extension edges
      (`-(2**31)`, `2**31 - 1`, `2**31`, `2**62`, `-2**62`),
      values straddling the 32-bit boundary (`2**32`, `2**32 - 1`,
      `2**63`, `2**63 + 1`), and the cycle-19 canonical reproducer
      value (`6_000_000_000`). The cross-class assertion
      `_wrap_int_to_type(v, isize) == _wrap_int_to_type(v, i64)`
      is the **cycle-3 alias-canon** the test docstring cites —
      this is the contract that was violated pre-fix and now
      pins it for regression. The assertion message includes both
      computed values so a future regression is self-diagnosing.
    - Uses `tir.TIRScalar(name="...")` to build the scalar types
      directly. Matches how the rest of the test file constructs
      types (e.g. `test_stage17_emits_mov_eax_14` at line 333:
      `tir.TIRScalar(name="i32")`). Idiomatic.
    - The test does NOT exercise the end-to-end compile + run
      path. This is appropriate — the cycle-19 audit's
      end-to-end reproducer is documented in the audit doc's §
      *Concrete reproducer* (lines 174-253 of
      `audit-stage28-8-cycle19-type-design.md`); the test-suite
      regression layer pins the unit contract, which is the
      necessary-and-sufficient condition for the end-to-end
      reproducer to also pass. The cycle-17 regression test
      (`test_c16_1_wide_array_elem_traps_at_codegen`) went
      end-to-end because that defect's manifestation required
      the full codegen pipeline (a trap raise); the cycle-20
      C19-1 defect's manifestation is at the unit level
      (`_wrap_int_to_type`'s value-in / value-out), so a unit
      test is the appropriately-scoped regression layer.

(h) **Test-suite verification**. Ran `python helixc/tests/
    test_const_fold.py` directly (in-process, no pytest harness):

    ```
    37 passed, 0 failed
    ```

    The commit message claims 37 const_fold tests pass (was 36
    cycle 17, +1 new). Verified at HEAD: 37 / 37 PASS, including
    `test_c19_1_isize_usize_are_64_bit_in_wrap` and the
    pre-existing `test_stage17_i32_overflow_wraps_two_complement`
    (which is the cycle-17 baseline test for `_wrap_int_to_type`'s
    32-bit wrap path on i32 — unchanged by the cycle-20 fix and
    still passing, confirming no regression to the narrow-int
    behaviour). Also re-ran `helixc/tests/test_typecheck.py`:
    **104 passed, 0 failed, 7 skipped** — no regression in the
    typecheck layer that supplies `res.ty` to the fold sites.

(i) **End-to-end adversarial probe** (read-only, in-process via
    the production toolchain). Per the user directive, ran the
    `fn main() -> isize { let a: isize = 3_000_000_000; let b:
    isize = 3_000_000_000; a + b }` reproducer at -O0 vs -O1.

    Probe harness (Python in-process, drives `parse →
    typecheck → lower → [fold_module if -O1] →
    compile_module_to_elf`; copies the ELF to WSL `/tmp/`,
    chmods +x, runs via `subprocess.run(['wsl', 'bash', '-c',
    '/tmp/probe'])`, captures `.returncode` directly — NOT via
    `echo $?` which masks the helix program's exit code with
    bash's `echo` exit code, as identified during this audit).

    Result on the user-suggested reproducer:

    | Variant | Exit code | Expected (6e9 mod 256) |
    |--------:|----------:|-----------------------:|
    | -O0 (no fold)  | **0** | 0 |
    | -O1 (with fold) | **0** | 0 |

    Both variants return 0. The exit codes agree. However, the
    user-suggested probe **cannot disambiguate the pre-fix from
    the post-fix state** at the exit-code level, because
    `6_000_000_000 mod 256 == 1_705_032_704 mod 256 == 0` — the
    low 8 bits are invariant under 32-bit-truncation by
    construction, and `wait()`/`returncode` only returns the low
    8 bits. (This is a property of unix exit codes, not of the
    fix; documenting here so future audits don't repeat the
    invariant probe.)

    Designed a **differentiating probe** that exposes the fix
    via a comparison that branches on the wide value:

    ```helix
    fn main() -> i32 {
        let a: isize = 3000000000_isize;
        let b: isize = 3000000000_isize;
        if a + b > 4000000000_isize { 17 } else { 23 }
    }
    ```

    The fold-pass folds `a + b` to a single `CONST_INT` carrying
    either `6_000_000_000` (post-fix, correct) or `1_705_032_704`
    (pre-fix, narrow-wrapped). The comparison `> 4_000_000_000`
    then evaluates true (post-fix, return 17) or false (pre-fix,
    return 23). Exit codes 17 and 23 fit in the low 8 bits and
    are unambiguously distinguishable.

    | Variant | Exit code | Pre-fix expected | Post-fix expected |
    |--------:|----------:|------------------|-------------------|
    | -O0 (no fold)  | **17** | 17 (runtime ADD is 64-bit) | 17 |
    | -O1 (with fold) | **17** | 23 (folded ADD wraps to 32-bit) | 17 |

    **Both variants return 17.** -O0 and -O1 agree. The folded
    path matches the un-folded path. The optimization-unstable
    miscompile of C19-1 is closed at runtime.

(j) **IR-level verification**. Dumped the lowered + folded IR
    for the user-suggested reproducer:

    ```
    CONST_INT {'value': 3000000000} res_ty= ["TIRScalar(name='isize')"]
    CONST_INT {'value': 3000000000} res_ty= ["TIRScalar(name='isize')"]
    CONST_INT {'value': 6000000000} res_ty= ["TIRScalar(name='isize')"]
    RETURN {} res_ty= []
    ```

    The folded ADD carries `value=6_000_000_000` (post-fix);
    pre-fix this would have carried `value=1_705_032_704` (the
    cycle-19 doc dumped exactly this trace at lines 249-253).
    The IR-level fix is observably present.

(k) **Cross-check for missed type-classifier sites with isize/usize
    width assumptions** (carryover audit from cycle 19 §
    *Cross-check: any third type-classifier site missed?*).
    Re-grepped at HEAD for `isize`/`usize`/`_BITS`/`width`
    patterns across `helixc/`:

    - `helixc/backend/x86_64.py:1011, 1017` — cycle-19 fix,
      isize/usize both classed as 64-bit. Correct.
    - `helixc/backend/x86_64.py:_check_array_elem_size_supported`
      — cycle-17 fix, `wide_widths = {"i64", "u64", "f64",
      "isize", "usize"}`. Correct.
    - `helixc/ir/passes/const_fold.py:46` — **cycle-20 fix, this
      audit**. Correct post-fix.
    - `helixc/frontend/typecheck.py:225-228, 241, 1816-1817` —
      isize→i64 and usize→u64 canon. Correct (pre-existing).
    - `helixc/frontend/autodiff.py:60-79` — broad NUMERIC_FOR_AD
      set, no width dispatch. Correct.
    - `helixc/frontend/lexer.py:90-91` — token-keyword routing
      only (`KW_ISIZE`, `KW_USIZE`), no width logic.
    - `helixc/frontend/monomorphize.py:704` — comment only.
    - `helixc/ir/lower_ast.py:357-358` — primitive name
      preservation, no width logic.
    - `helixc/backend/ptx.py:328-332` — cycle-18 forward note
      already filed; no isize/usize entries in dtype-suffix map;
      KeyError on isize-tensor dtype is loud, not silent. Phase-0
      narrow+loud pattern. Not a regression of C19-1's class.
      Remains as a forward note.
    - `helixc/ir/passes/cse.py`, `helixc/ir/passes/dce.py`,
      `helixc/ir/passes/fdce.py`, `helixc/ir/passes/effect_check
      .py` — re-grepped: zero references to isize/usize/_BITS/
      width/bits. None of these passes classify by width. No
      latent fourth site.

    **Confirmed: no fourth site**. The cycle-19 cross-check
    enumeration remains complete with the cycle-20 fix landing.

(l) Cross-checked the docstring on `_wrap_int_to_type` (lines
    59-65). Pre-existing docstring documents the function's
    intent ("two's-complement, like x86 hardware") and cross-
    references the runtime ADD path. No update to the docstring
    was required because the function's contract (input → wrap →
    output) is unchanged; only the underlying `_INT_BITS`
    lookup widened. The docstring stays correct.

(m) Reviewed the diff for unrelated drift. The non-doc files
    in the commit are exactly:
    - `helixc/ir/passes/const_fold.py` (+9 / -1 lines: width
      edit + 7-line audit-stamp comment).
    - `helixc/tests/test_const_fold.py` (+31 lines: one new
      test function).

    The one md file added is `docs/audit-stage28-8-cycle19-type-
    design.md` (the cycle-19 type-design audit doc being persisted
    alongside its fix — appropriate). No production code was
    modified outside the `_INT_BITS` entries. No spurious
    whitespace / import changes.

(n) Cycle-counter accounting. The cycle-19 type-design audit
    found C19-1 (HIGH) and reset the counter to 0/5. The cycle-
    20 fix-sweep closes C19-1 — but does NOT itself constitute a
    clean cycle; it constitutes the *closure event* whose
    verification produces cycle 20's three audits. Cycle 20's
    three audits (silent-failures, type-design, code-review)
    collectively decide whether to advance to 1/5. This code-
    review audit's verdict is one of the three required CLEAN
    votes for that advance.

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

The probes exercise the cycle-20 fix's correctness across the
const-fold result-wrap surface and the end-to-end compile +
run path. Cycle 19 surfaced C19-1 from a single arithmetic
reproducer (`3e9 + 3e9 = 6e9` isize); cycle 20's probes extend
coverage:

1. **User-suggested exit-code probe** (`a + b`, return isize):
   exit codes agree at -O0 and -O1 (both 0). But the exit-code
   level is **insensitive** to 32-vs-64-bit truncation when the
   wide value's low 8 bits coincide between the truncated and
   un-truncated forms (which is the case here: `6e9 mod 256 ==
   1.7e9 mod 256 == 0`). Useful as a smoke probe; cannot
   *demonstrate* the fix.

2. **Differentiating-comparison probe** (`if a + b > 4e9 { 17 }
   else { 23 }`, return i32): exit codes 17 (post-fix) vs 23
   (pre-fix). Result at HEAD: -O0 returns 17, -O1 returns 17,
   they agree. Pre-fix -O1 would have returned 23. **This
   probe positively demonstrates the fix at the ELF execution
   level.**

3. **IR-level value check** (folded CONST_INT value): at HEAD
   the folded ADD carries `value=6_000_000_000`, type isize.
   Pre-fix it would carry `value=1_705_032_704`. **Direct
   inspection of the const-folded output confirms the fix.**

4. **Unit-level cross-class round-trip** (15 values across
   isize/i64 + usize/u64 in the regression test): all pairs
   agree. The fix's coverage extends across the full
   sign-extension boundary, the 32-bit boundary, and the
   typical large-pointer values.

The most adversarial of the set:

- **Differentiating-comparison probe (probe 2)**: distinguishes
  pre-fix from post-fix at the runtime-exit-code level, which
  the user-suggested probe alone cannot. It also confirms the
  fix is **optimization-stable** (-O0 == -O1 == 17), which was
  the principal cycle-19 complaint ("Wrong AND optimization-
  unstable — the program's behavior depends on whether -O1 fires").

- **Unit-level cross-class round-trip (probe 4)**: pins the
  cycle-3 alias-canon invariant in the test suite for all
  future audits. The same canon was the cycle-3 C3-2 finding
  at the widening-rank level; the cycle-20 fix completes the
  canon at the const-fold layer. Future regressions of this
  class will trip the unit test immediately.

---

## Why no false-positive risk

1. **Fix is mechanically minimal**: a one-line `_INT_BITS` edit
   from 32 → 64 for two keys. The function body and all four
   call sites are unchanged. The blast radius is exactly the
   set of const-fold ops whose result type is `TIRScalar
   ("isize")` or `TIRScalar("usize")` — a narrow, well-defined
   surface that the cycle-19 audit traced fully.

2. **Test pins the proof-of-fix value**: the regression test's
   first assertion is `_wrap_int_to_type(6_000_000_000, isize)
   == 6_000_000_000`. Pre-fix this would fail (yields
   1_705_032_704); post-fix it passes. The test is a direct
   regression gate for the exact reproducer cycle 19
   documented.

3. **Cross-class agreement is enforced**: the test verifies
   `_wrap_int_to_type(v, isize) == _wrap_int_to_type(v, i64)`
   over a representative range of values. This pins the alias
   contract (cycle-3 C3-2 canon) at the const-fold layer in a
   way that any future divergence will catch.

4. **No regression in narrow-int paths**: `_INT_BITS` entries
   for i8, u8, i16, u16, i32, u32, bool are unchanged. The
   cycle-17 baseline test `test_stage17_i32_overflow_wraps_two_
   complement` passes (32-bit two's-complement wrap on i32 is
   unchanged). The `bool` entry (32, "bool comparisons reified
   to i32 in IR") is preserved with its original comment.

5. **No regression in i64/u64 paths**: pre-fix and post-fix
   both class i64 / u64 as 64-bit; only the isize / usize
   aliases changed. Any test exercising raw i64 / u64 const-
   fold is unaffected.

6. **End-to-end probe confirms runtime behavior**: the
   differentiating-comparison probe at -O0 and -O1 both
   return 17 (the post-fix expected value). The folded path
   now agrees with the un-folded path. Optimization stability
   is restored.

7. **Cycle-19 cross-check ruled out a fourth site**: the
   cycle-19 audit explicitly enumerated every type-classifier
   site in `helixc/` and confirmed const_fold's `_INT_BITS`
   was the only outstanding one. Re-grep at HEAD confirms no
   new classifier sites have been introduced; the cycle-20
   fix is the complete close.

8. **The 5-step pipeline trace from cycle 19 is verified
   end-to-end at HEAD**: I re-ran the cycle-19 doc's pipeline
   trace (parse → typecheck → lower → fold_module) and
   confirmed the folded `CONST_INT` now carries
   `value=6_000_000_000` (pre-fix: `1_705_032_704`). Direct
   IR-level confirmation of the fix.

---

## Below-threshold observations from this cycle

### B20-1 — `_wrap_int_to_type` sign-extends unsigned types (conf 60, observation, pre-existing)

**Location**: `helixc/ir/passes/const_fold.py:59-74`.

```python
def _wrap_int_to_type(value: int, ty: "tir.TIRType") -> int:
    ...
    bits = 32  # default for unknown / generic scalar types
    if isinstance(ty, tir.TIRScalar):
        bits = _INT_BITS.get(ty.name, 32)
    mask = (1 << bits) - 1
    half = 1 << (bits - 1)
    v = value & mask
    if v >= half:
        v -= (1 << bits)
    return v
```

The wrap function applies the same sign-extension correction
regardless of signedness: `if v >= half: v -= (1 << bits)`. For
a `u32` value of `0xFFFFFFFF` (4_294_967_295), this returns
`-1` (Python int), not `4_294_967_295`. Same for u64, usize:
`_wrap_int_to_type(2**63, usize)` returns
`-9_223_372_036_854_775_808` instead of
`9_223_372_036_854_775_808`.

This is **a pre-existing behaviour, not introduced by C19-1's
fix**. The const_fold contract appears to be that all wrapped
values are stored as Python ints in two's-complement-signed
form, and the downstream backend treats the stored value's
*bit pattern* (not its sign) as the canonical representation.
This works because `mov rax, imm64` doesn't care about signed
vs unsigned at the bit level. But it would surprise a casual
reader of `_wrap_int_to_type`, and an unsigned-comparison fold
that consumed the result as an unsigned-Python-int could
misbehave.

Did NOT promote to a finding because:
- The pre-existing tests (e.g. `test_stage17_i32_overflow_wraps_
  two_complement` on -1 / `0xFFFFFFFF`) pass, suggesting the
  convention is intentional and the downstream consumers
  honor it.
- No reachable miscompile path identified within the cycle-20
  audit window (would require constructing a u32/u64/usize
  const-folded value that flows into an unsigned comparison
  that const-folds on the Python sign rather than the bit
  pattern — the cycle-19 comparison-fold path at lines 417-441
  does its own per-op comparison logic and does not call
  `_wrap_int_to_type` on the comparison result, so the sign-
  extended value does not directly leak into the comparison
  result).
- Pre-existing behaviour, not a cycle-20 regression.

Recorded for Stage-29 attention as part of the "centralize
the scalar-width-and-signedness predicate" Stage-29-class
refactor that the cycle-17 / cycle-18 / cycle-19 docs all
forward-flagged. A proper scalar-width-and-sign predicate
would distinguish "wrap to signed Nbits" from "wrap to
unsigned Nbits" and call the correct one based on
`_is_signed_type(ty)`. Currently `_wrap_int_to_type` does the
signed variant for all types.

Confidence 60: real observation, but no reachable miscompile
demonstrated within this audit window, and behaviour is
pre-existing (not introduced by C19-1's fix or any cycle
since).

### Carryover from prior cycles (unchanged)

- **B17-2** (LOAD_ELEM `op.operands[0].ty` not checked) —
  unchanged at HEAD (cycle 17 → 18 → 19 → 20 production-code
  delta in this region is empty). Remains conf 35.
- **B17-3, B17-4** — unchanged.
- **B18-1** (tuple literal lowering shares the same decay
  defect as C18-1) — note that C18-1 was closed in cycle 19's
  fix-sweep (commit 0803902); whether the parallel tuple
  fast-path at `lower_ast.py:885-906` got the same fix is
  noted as a Stage-29 follow-up in the cycle-19 doc. Not
  re-examined here (out of scope for the cycle-20 fix-sweep
  audit).
- **B18-2** (typechecker Index returns unconditional TyUnknown)
  — Stage-29-class type-inference completeness, unchanged.
- **Cycle-18 / cycle-19 forward notes** — PTX dtype-suffix
  map missing isize/usize entries (loud KeyError, not silent;
  not a C19-1-class issue); centralize scalar-width-and-sign
  predicate Stage-29 refactor.
- **Earlier carryovers** (B10-x, B14-2, B15-1, cycle-16
  forward notes) — unchanged, remain Stage-29-class.

---

## Cycle 20 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This audit (Audit C, code-review) finds **0 findings at confidence
≥ 80.**

**Counter status (5-clean-consecutive gate)**:
- Was 0/5 after cycle 19 reset.
- Cycle 20 code-review (this audit): CLEAN under the strict
  criterion. Provisional advance to 1/5 conditional on cycle
  20's other two audits (silent-failures, type-design) also
  voting CLEAN.
- The cycle-19 carryover C19-1 (HIGH, conf ≥ 95) is closed by
  the cycle-20 fix-sweep at commit 5a1e406. Verified the
  closure here:
  - The one-line `_INT_BITS` edit is correctly placed.
  - All four `_wrap_int_to_type` call sites correctly route
    through the fixed table.
  - The regression test exercises both classes (isize/i64 +
    usize/u64) across 15 representative values.
  - End-to-end differentiating probe at -O0 and -O1 both
    return the post-fix expected value (17). Optimization
    stability restored.
  - IR-level dump shows folded ADD carries 6_000_000_000
    (post-fix), not 1_705_032_704 (pre-fix).
  - No fourth type-classifier site has been missed; cycle-19
    cross-check is verified complete at HEAD.

The severity trend across cycles, against the strict-criterion bar:
- Cycle 1: HIGH-tier — not clean
- Cycle 2: HIGH + MEDIUM — not clean
- Cycle 3: HIGH + MEDIUM + LOW — not clean
- Cycle 4: MEDIUM — not clean
- Cycle 5: 3 MEDIUM + 3 LOW — not clean
- Cycle 6: 1 MEDIUM + 2 LOW — not clean
- Cycles 7-12: 0 + 0 + 0 — clean (counter advanced to 3/5)
- Cycle 13: 1 HIGH (C13-1) — not clean → reset to 0/5
- Cycle 14: 0 + 0 + 0 — clean → 1/5
- Cycle 15: 0 + 0 + 0 — clean → 2/5
- Cycle 16: 1 HIGH (C16-1) — not clean → reset to 0/5
- Cycle 17: 0 + 0 + 0 — clean → 1/5
- Cycle 18: 1 HIGH (C18-1) — not clean → reset to 0/5
- Cycle 19: 1 HIGH (C19-1) — not clean → reset to 0/5
- Cycle 20 code-review (this audit): 0 findings — CLEAN.
  Provisional advance 0/5 → 1/5 pending silent-failures and
  type-design audits.

---

## Verdict

**CLEAN** under Audit C (code-review) at HEAD (5a1e406).

The cycle-20 fix-sweep correctly closes C19-1 (HIGH). The
one-line `_INT_BITS` edit at `helixc/ir/passes/const_fold.py:53`
changes `isize` / `usize` from 32 → 64 bits, matching
`typecheck.py:225-228`'s widen-canon and the cycle-19 backend
classifier fix at `x86_64.py:1005-1017`. All four
`_wrap_int_to_type` call sites in const_fold (ADD/SUB/MUL/DIV/MOD
result wrap, bitwise/shift result wrap, NEG, BIT_NOT) correctly
route through the fixed table; the fix has uniform reach across
the const-fold result-wrap surface.

The new regression test
`test_c19_1_isize_usize_are_64_bit_in_wrap` pins the cycle-3
alias-canon at the const-fold layer for both classes:
isize/i64 agreement and usize/u64 agreement, across 15
representative values including sign-extension edges and the
canonical cycle-19 reproducer value (6_000_000_000). 37
const_fold tests pass at HEAD (was 36 cycle 17 → +1 new).

The user-suggested adversarial probe (`a + b` returning
isize, -O0 vs -O1) was found to be **insensitive to the bug at
the exit-code level** (because `6e9 mod 256 == 1.7e9 mod 256
== 0` — the low 8 bits coincide). A **differentiating
comparison probe** (`if a + b > 4e9 { 17 } else { 23 }`
returning i32) confirms the fix end-to-end: both -O0 and -O1
return 17 (post-fix expected); pre-fix -O1 would have returned
23. Optimization stability is restored.

No fourth type-classifier site has been missed. The cycle-19
cross-check enumeration is re-verified complete at HEAD.

One pre-existing below-threshold observation logged (B20-1):
`_wrap_int_to_type` applies signed two's-complement
wrap-and-sign-extend regardless of the type's signedness. Not
introduced by the cycle-20 fix; appropriate as a Stage-29
"centralize scalar-width-and-sign predicate" refactor item.
Conf 60.

Cycle counter: **provisional 1/5** under this audit (pending
silent-failures and type-design verdicts to confirm).

Forwarded for future-cycle attention: B20-1 (unsigned-wrap
sign-extension); B17-2/3/4, B18-1, B18-2 carryovers; PTX
isize/usize dtype-suffix forward note (cycle 19);
centralize-scalar-width predicate Stage-29 refactor.
