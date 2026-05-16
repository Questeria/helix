# Stage 36 Post-Increment-8 Audit — Code-Review Lane

**Date**: 2026-05-16
**Auditor**: pr-review-toolkit:code-reviewer
**HEAD audited**: `a451591` (Stage 36 Increment 8)
**Baseline**: `b8cafe7` (Stage 35 closure)
**Status**: **0 HIGH + 3 MEDIUM + 1 LOW** (chain-rule math and dogfood SGD math correct)

## AD chain-rule math correctness (all verified vs spec)

- `fuzzy_and` forward (`a'*b + a*b'`) and reverse (`adj*b`, `adj*a`) — correct
- `fuzzy_or` forward and reverse (`1-b`, `1-a`) — correct
- `fuzzy_not` forward (`-a'`) and reverse (`-adj`) — correct
- `fuzzy_xor` (`1-2b`, `1-2a`) — correct
- `fuzzy_implies` (`-1+b`, `a`) — correct

Dogfood SGD math (07 closed-form `w → 0.8` in 1 step at `lr=2.0`; 08
`w_new = target` in 1 step at `lr=0.5`) verified algebraically.
Exit-42 assertions are mathematically sound.

## Findings

### B1 MEDIUM (conf 82) — `derive(a, b)` lowering swaps source-order evaluation

**File**: `helixc/ir/lower_ast.py` (Inc 2 `derive` handler)

Code evaluates `expr.args[1]` first ("for side effects") then returns
the lowering of `expr.args[0]`. Source order is `derive(a, b)` — a
user reading the program expects `a` then `b`. Helix functions can
have observable side effects (io::println etc.), so
`derive(log("a"), log("b"))` would print `b` before `a`.

**Fix**: lower `args[0]` first, then `args[1]`, then return the
first.

### B2 MEDIUM (conf 88) — No reverse-mode tests for `∂/∂b` of any 2-arg fuzzy op

**File**: `helixc/tests/test_stage36_provenance.py:644-802`

Every numerical AD test fixes the second argument as a *literal*
(`prove(0.5_f32, 0)`) and differentiates only against the first
parameter `x`. Reverse-mode bugs that flip `a_arg`/`b_arg` or use
the wrong `coeff_a`/`coeff_b` would not be caught — the second-arg
path has zero coverage. Given the asymmetric formulas of `fuzzy_or`
(`1-b` vs `1-a`) and `fuzzy_implies` (`-1+b` vs `a`), a transpose
bug is genuinely possible.

**Fix**: add at least one test per op where the differentiated
variable is the *second* arg, e.g.
`loss(x) = unwrap_logic(fuzzy_implies(prove(0.3_f32, 0), prove(x, 0)))`
and assert `grad_rev(loss)(0.5) == 0.3`.

### B3 MEDIUM (conf 80) — No finite-difference cross-check

**File**: `helixc/tests/test_stage36_provenance.py`

All tests compare against analytic expected values only. A test
computing `(loss(x+h) - loss(x-h)) / (2h)` and comparing to
`grad_rev(loss)(x)` inside Helix would be a stronger contract.

**Fix**: add one nested-composition test per mode (forward/reverse)
that does central-difference inside Helix and asserts the two are
within tolerance.

### C1 LOW (conf 70) — `derive`, `*_logic` integer ops missing from `AD_KNOWN_PURE_CALLS`

**File**: `helixc/frontend/autodiff.py:47-68`

`derive`, `and_logic`, `or_logic`, `not_logic`, `xor_logic`,
`implies_logic`, `eq_logic`, `if_logic`, `to_logic_bool`,
`register_derivation`, `parent_left_at`, `parent_right_at` are
absent. They are integer-valued so non-differentiable, and reverse-
mode now fails closed on opaque calls — so any `grad()`/`grad_rev()`
over a function touching these will reject. Likely intentional but
undocumented.

## Verdict

No HIGH-severity findings. The chain-rule math is correct in both
modes; the dogfood SGD math is sound; the typecheck/lower/AD
registration matrix is consistent across the eight Inc-1-to-8
fuzzy/logic ops. The two MEDIUM items (`derive` evaluation order +
`∂/∂b` test coverage) are worth fixing in the next increment but do
not block Stage 36 closure.
