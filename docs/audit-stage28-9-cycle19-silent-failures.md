# Audit Stage 28.9 cycle 19 — Silent failures

**Scope.** Read-only HEAD `46e9952`. Prior C1–C18 not re-flagged.
**Criterion.** 0 findings at conf >=75%.

## Result: 1 finding at >=75% — FAIL

## Finding C19-1 — const-fold drops out-of-range shift silently

**Severity:** HIGH. **Confidence:** 78.
**Location:** `helixc/ir/passes/const_fold.py` lines 409-416.

**Issue.** SHL/SHR fold returns `None` when `r < 0 or r >= 64`
("UB; leave as runtime op"). No diagnostic. typecheck has no
shift-range guard. `1_i32 << 64` (both CONST_INT) compiles cleanly;
x86 SHL masks count → `shl eax, 64` = `shl eax, 0` (no-op).
Refusing fold is correct; refusing silently when both operands are
statically known is the silent-failure pattern.

**Hidden errors.** Compile-time-known shifts past type width produce
silent runtime no-ops. Asymmetric to FoldError NaN at 374/472, which
raises trap-17001 for compile-time NaN folds.

**Impact.** "Why does `1_i32 << 32` evaluate to 1, not 0?" — no
diag, no trap-id.

**Recommendation.** Raise `FoldError` (trap 17002) when const shift
is outside `[0, bit_width)`. `_INT_BITS[res.ty.name]` exposes width.

## Notes (<75)
C18-C1 cse comment still references non-existent
`_propagate_identities` list-copy (cycle 18, ~70).
