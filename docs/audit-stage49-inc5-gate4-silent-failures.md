# Stage 49 Inc 5 — Gate-4 Silent-Failure VERIFICATION Audit

**Date:** 2026-05-17
**HEAD:** `b4c8434` (gate-3 G3-H1 map_ok/map_err wider-payload reject)
**Method:** read-only verification of gate-1/2/3 fix landings + cascading-defect probes.
**Baselines:** `docs/audit-stage49-inc5-gate1-silent-failures.md`.

## VERDICT: FIXES REQUIRED — gate-4 cycle must be reset.

One NEW CRITICAL silent-miscompile surface (nested Result) is open at `helixc/frontend/typecheck.py:6417-6419`. The gate-2 G2-H1 fix introduced an over-permissive whitelist that cascade-propagates the original SF1-F2 defect class to a sibling surface. Gate-3's G3-H1 fix reuses the same helper and therefore inherits the same defect.

## Summary

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| SF4-C1 | CRITICAL | 96 | Nested `Result<Result<T,E1>, E2>` silently miscompiles — `_reject_non_i32_result_payload` whitelists TyResult |
| SF4-C2 | CRITICAL | 95 | `map_ok(r, Result-value)` / `map_err(r, Result-value)` silently miscompile (same helper, different entry — G3-H1 inherits the bug) |
| SF4-V1 | n/a | 99 | SF1-F2 (i32-only constructor reject) — CLOSED at `typecheck.py:4557-4572` |
| SF4-V2 | n/a | 99 | SF1-F1 (`__try` Err arm TRACE_EXIT) — CLOSED at `lower_ast.py:2204-2208` |
| SF4-V3 | n/a | 95 | gate-3 G3-H1 (map_ok/map_err wider primitive payload) — CLOSED at `typecheck.py:4862,4898` |

## SF4-C1 — CRITICAL (96): Nested Result silent miscompile

**Location:** `helixc/frontend/typecheck.py:6417-6419`

```python
# Accept nested Result (identity-recurses to i64 packed).
if isinstance(stripped, TyResult):
    return
```

**Defect class:** same as SF1-F2 (width truncation at RESULT_PACK 32-bit payload slot), at a new entry. The gate-2 G2-H1 helper explicitly whitelists TyResult — but `_lower_type(Result<T,E>)` returns `TIRScalar("i64")`, so a nested Result IS i64-wide and IS exactly the case the helper was supposed to reject.

**Repro (verified live against HEAD):**

```helix
fn make_inner() -> Result<i32, i32> { Err(99) }
fn make_nested() -> Result<Result<i32,i32>, i32> { Ok(make_inner()) }
fn main() -> i32 {
    let r = make_nested();
    let inner = unwrap_ok(r);
    unwrap_err(inner)   // expected: 99; actual: truncated payload
}
```

Result: `typecheck errs=0; lower OK; compile OK (elf size 406893)`. The Inc 1.5 wrong-arm runtime trap does NOT fire (the outer `unwrap_ok` is structurally Ok-correct); the inner Err's high-32 bits — including the tag bit at position 32 — are silently dropped by the 4-byte `mov ecx, [rbp+...]` at `x86_64.py:2205`. The "inner" Err(99) decodes as Ok(99) or worse — true silent miscompile.

**Hidden errors:** any nested Result composition; any user library that wraps `Result<T,E>` inside another `Result<*, E2>` (the standard Rust-style "result-of-result" for layered error handling); any inferred-nested case via `map_ok` (see SF4-C2).

**User impact:** identical to SF1-F2 — the typecheck lies. Source looks fine, typecheck passes, tests pass (no test uses nested Result), the bytes vanish.

**Recommendation:** remove the TyResult whitelist at `typecheck.py:6417-6419`. Reject nested Result with the same diagnostic vocabulary as the i64/f64/struct cases. If nested Result is genuinely required for Phase-0, defer it to Stage 50+ alongside the wider-payload widening.

## SF4-C2 — CRITICAL (95): map_ok / map_err Result-valued new_value silently miscompile

**Location:** `helixc/frontend/typecheck.py:4862` and `:4898` (G3-H1 fix sites). The reject calls `_reject_non_i32_result_payload(arg_tys[1], …)`, which whitelists TyResult (SF4-C1).

**Repro:** `let r2 = map_ok(r, make_inner())` where `make_inner` returns `Result<i32,i32>` typechecks with 0 errs. The SELECT then packs an i64 new_value into a 32-bit payload slot — high 32 bits silently dropped, including the inner tag.

**Why gate-3 missed this:** G3-H1 correctly identified that map_ok/map_err were not calling the G2-H1 helper. It wired the call, but inherited the helper's TyResult-whitelist bug from SF4-C1. The defect cascade is one step deeper than gate-3 looked.

**Recommendation:** co-fix with SF4-C1 (single helper change closes both).

## Verified closed (audit hygiene)

- **SF1-F2 (i32 payload reject):** live-probed `Result<i64,i64>`, `Result<f64,i32>`, `Result<f32,i32>`, `Result<bool,i32>`, `Result<char,i32>`, `Result<u32,i32>` — all reject with the canonical G2-H1 diagnostic. Reject IS appropriately conservative (4-byte `mov ecx` load would garbage-extend 1/2-byte slots; `mov_mem_rbp_eax` 4-byte store would clobber adjacent slots for sub-i32 types). No existing test or example uses these forms — over-narrow reject does not break valid programs.
- **SF1-F1 (`__try` Err arm TRACE_EXIT):** IR-inspected `@trace fn helper(): r? ; Ok(v+1)` — both early-return blocks (Err-propagation block id=650; Ok-return block id=651) emit `trace.exit` immediately before `return`. Pin-test `test_stage49_gate2_sf1f1_try_err_arm_emits_trace_exit_in_traced_fn` + sanity sibling `test_stage49_gate2_sf1f1_untraced_fn_with_try_still_no_trace_exit` cover both arms.
- **gate-3 G3-H1 (map_ok/map_err primitive wider payload):** rejects i64/f64 new_value with the "map_ok new_value" / "map_err new_value" diagnostic. Pin-tests at lines 984, 1009.

## Cascading-defect rhythm

Gate-2 introduced `_reject_non_i32_result_payload` to close SF1-F2. Gate-3 wired map_ok/map_err to the helper to close G3-H1. Gate-4 finds the helper's own whitelist (line 6418) leaks the same defect class one layer deeper. This is the **exact** rhythm the gate-1 audit warned about: "every Stage closure gate finds a HIGH at a NEWLY-introduced layering seam … the discipline exists, the carry-forward to new sites was missed." Here the missed site is the helper's own permissive arm.

The fix is one line (delete the TyResult whitelist) plus one regression test. Both SF4-C1 and SF4-C2 close together.

## Self-host + suite posture

- `pytest helixc/tests/test_stage49_runtime_tag.py -q` — **48 passed in 155s.**
- `pytest helixc/tests/test_stage46_result.py -q` — **27 passed in 67s.**
- `pytest helixc/tests/test_panic.py -q` — **25 passed in 8s.**
- dogfood_17 exit 42 covered by `test_stage49_inc4_dogfood_17_still_exits_42` (in the 48-pass).

Suite is green; the silent miscompiles are uncovered (no test uses nested Result or map_ok with a Result-typed new_value).

## Cross-lane confirmation

The parallel type-design gate-4 lane (`docs/audit-stage49-inc5-gate4-type-design.md`) independently identified the same nested-Result defect as TD1-C2 + TD4-C3 with the same root cause. Two lanes converging on one root strengthens confidence to 96/95.

## Recommendation

Reset gate-4. Land SF4-C1 + SF4-C2 (single helper edit + 2 regression tests covering nested Result and map_ok-of-Result-value), then rerun gate-4 verification.
