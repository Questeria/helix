# WIP BLOCKER — test_codegen.py failures (8 of 9 RESOLVED)

**Filed:** 2026-05-20 ~02:05 UTC by helix-approach-a-loop fire.
**HEAD when found:** `8564e6f` (v2.5 audit-fix line).
**Status:** Bucket A (8 tests) **RESOLVED** 2026-05-20 — all 8 were
pre-Cycle-3-contract stale tests, fixed this fire (test-only edits).
Only **Bucket B** (1 test, `test_hbs_sample_tree_eval_runs`) remains
open — a distinct root cause; see that section below.

## Bucket A resolution (2026-05-20)

All 8 Bucket A tests were confirmed STALE — same class as `e7768f4`:
Cycle 3 audit batches deliberately changed the stdlib guard
contracts, and these tests still encoded the pre-Cycle-3 contracts.
Per-test:

- `log_f64_domain_guard` — `__log_stable_f64(x<=0)` returns NaN now
  (Cycle 3 R2 batch 25), not the old -1e6 sentinel.
- `vec_zip_div_zero_divisor_fail_closed` — a zero divisor yields the
  INT32_MIN sentinel now (Cycle 3 R1 batch 20), not 0.
- `vec_l2_squared_distance_saturates` — the function clamps each
  delta to [-46340, 46340] before squaring now (Cycle 3 R1 batch 20),
  so one element can't overflow; the test uses 2 elements to exercise
  the accumulator-saturation path it was always meant to check.
- the 5 `ce_loss` / `ce_loss_batch_f32` tests — both return NaN
  (0.0/0.0) for an invalid label now (Cycle 3 R1 batch 20), not a
  large finite value; tests switched to the `loss != loss` NaN idiom.

Fix = test-only edits (the stdlib code is correct). Verified: 8 pass.

## How this was found

A full `python -m pytest helixc/tests/test_codegen.py` run (1h43m,
1021 passed / 10 failed) surfaced these. They went unnoticed because
the full `test_codegen.py` suite is too slow to run per-fire, so fires
run subsets (`test_typecheck.py`, `test_ptx.py`, etc.). One of the
original 10 — `test_stage35_wmt_predictors_reject_invalid_and_corrupt_states`
— was already fixed by concurrent fire `e7768f4` as a **stale test**
("pre-Cycle-3 contract"). The remaining **9 still fail on `8564e6f`**.

## Reproduce

```bash
cd C:/Projects/Kovostov-Native
python -m pytest \
  "helixc/tests/test_codegen.py::test_nn_ce_loss_batch_f32_rejects_invalid_label" \
  "helixc/tests/test_codegen.py::test_nn_ce_loss_batch_f32_invalid_label_not_averaged_down" \
  "helixc/tests/test_codegen.py::test_stage35_nn_classifier_helpers_reject_short_outputs_and_targets" \
  "helixc/tests/test_codegen.py::test_nn_ce_loss_rejects_negative_scalar_label" \
  "helixc/tests/test_codegen.py::test_nn_ce_loss_rejects_positive_out_of_range_label" \
  "helixc/tests/test_codegen.py::test_hbs_sample_tree_eval_runs" \
  "helixc/tests/test_codegen.py::test_stage35_restart51_log_f64_domain_guard" \
  "helixc/tests/test_codegen.py::test_stage35_restart51_vec_zip_div_zero_divisor_fail_closed" \
  "helixc/tests/test_codegen.py::test_stage35_restart54_vec_l2_squared_distance_saturates" \
  --tb=short -q
```

## Bucket A — 8 stdlib-guard failures (`expected 42, got 7`/`1`) — RESOLVED (see top)

All 8 compile a program that calls a **stdlib (.hx) function** whose
guard / saturation / rejection branch is expected to fire (returns 42)
but instead the program falls through to the else branch (7 or 1).

| Test | assert line | got | stdlib fn under test |
|------|-------------|-----|----------------------|
| `test_nn_ce_loss_batch_f32_rejects_invalid_label` | 10007 | 1 | `ce_loss_batch_f32` invalid-label sentinel |
| `test_nn_ce_loss_batch_f32_invalid_label_not_averaged_down` | 10027 | 1 | `ce_loss_batch_f32` multi-row sentinel |
| `test_stage35_nn_classifier_helpers_reject_short_outputs_and_targets` | 10047 | 7 | nn classifier helper length guard |
| `test_nn_ce_loss_rejects_negative_scalar_label` | 10560 | 7 | `ce_loss` negative-label guard |
| `test_nn_ce_loss_rejects_positive_out_of_range_label` | 10577 | 7 | `ce_loss` out-of-range-label guard |
| `test_stage35_restart51_log_f64_domain_guard` | 21172 | 7 | `log_f64` domain guard |
| `test_stage35_restart51_vec_zip_div_zero_divisor_fail_closed` | 21245 | 7 | `vec_zip_div` div-by-zero fail-closed |
| `test_stage35_restart54_vec_l2_squared_distance_saturates` | 21656 | 7 | `vec_l2_squared_distance` i32 saturation |

The uniform symptom (guard never fires) across 8 unrelated stdlib
functions points to **one shared root cause** — most likely either:

1. **A contract change** in Cycle 3 / Cycle 5 audit fix batches (cf.
   `5b14ea1` "Cycle 3 fix batches 20-24" touched both test_codegen.py
   AND stdlib; `f8ededc` "Cycle 5 R1" touched stdlib). If the
   sentinel/guard *contract* changed, these are **stale tests** —
   same class as `e7768f4`. Fix = update the tests.
2. **A real codegen regression** in a shared primitive these guards
   depend on — i64 accumulator handling, INT32 saturation, or
   comparison codegen (`==` against a boundary constant such as
   `2147483647`). Fix = fix codegen.

**Triage method:** for each test, read the stdlib `.hx` definition of
the function it calls. If the function's current guard contract no
longer returns the value the test expects → stale test, update it. If
the contract is unchanged but the compiled program misbehaves → real
codegen regression; bisect across the Cycle 3 R1–R7 / Cycle 5 batches.

## Bucket B — 1 lowering NotImplementedError

`test_hbs_sample_tree_eval_runs` — distinct root cause:

```
NotImplementedError: lower_ast: A.Index on non-tensor/tile callee at
Span(line=24, col=48) reached lowering; typecheck should have rejected
this (Cycle 1 Batch IR silent-failure HIGH-3 fix)
  helixc/ir/lower_ast.py:4734
```

The test program indexes a non-tensor/non-tile callee. The lowering
pass (`lower_ast.py:4734`) correctly refuses it, but the message says
**typecheck should have caught it first**. Either (a) the typecheck
pass has a gap for this construct, or (b) the test program is itself
invalid and the test is stale. Read the `test_hbs_sample_tree_eval_runs`
source (~line 12xxx in test_codegen.py) at `Span(line=24, col=48)` to
decide.

## Process note (root cause of *late discovery*)

`test_codegen.py` takes **1h43m** for a full run, so no per-fire run
exercises it whole — regressions/stale-tests accumulate silently.
Recommend one of: (a) a periodic "full-suite" scheduled fire that runs
`test_codegen.py` end-to-end and files blockers, or (b) split
`test_codegen.py` into faster shards, or (c) a fast smoke subset gated
per-fire. This is separate from the 9 failures but is *why* they sat
unseen.

## Not done this fire / why

This fire is an overlapping fire on a 12-min cadence (concurrent fires
are actively committing). Triaging 9 tests across 2+ root causes —
including a possible git-bisect — exceeds one fire and a bisect would
require `git checkout` of old commits, which is unsafe while concurrent
fires share the working tree. Filed as a blocker for a focused fire (or
the user) to pick up.
