# Stage 49 Inc 5 closure gate-4 type-design VERIFICATION audit

**HEAD**: `b4c8434` (Stage 49 closure gate-3 G3-H1 fix: map_ok/map_err wider-payload reject)
**Gate-1 baseline**: `docs/audit-stage49-inc5-gate1-type-design.md` (2 CRITICAL + 1 HIGH + tail)
**Date**: 2026-05-17
**Method**: read-only. Re-ran gate-1's live probes against HEAD post-gate-1/2/3 fix cascade; targeted new probes against composition surfaces (nested Result via `map_ok`/`map_err`; tuple-of-Result; array-of-Result; ABI cross-check; wrapper-quintet end-to-end).

## VERDICT: FIXES REQUIRED — gate-4 cycle must be reset.

## Summary

| ID | Sev | Conf | Closed? | Title |
|----|-----|------|---------|-------|
| TD1-C1 | CRITICAL | 98 | YES (gate-2 G2-H1) | `Result<i64, *>` and `Result<*, i64>` silent truncation |
| **TD1-C2** | CRITICAL | 98 | **NO** | Nested `Result<Result<T,E1>, E2>` STILL silently miscompiles |
| TD4-C3 | CRITICAL | 95 | NEW | `map_ok(r, Result_value)` / `map_err(r, Result_value)` STILL silently miscompile (same root cause as TD1-C2, different entry) |
| TD1-H1 | HIGH | 92 | NO | Aggregate-of-Result (struct/tuple/array) still typecheck-clean → misleading backend error |
| TD1-MH1 | MED-HIGH | 85 | UNVERIFIED | static-prov C1 retirement plan-vs-code drift (not in this lane's scope) |
| TD1-M1/M2/M3 | MED | n/a | YES (gate-1/2 sweep) | comment/dead-code polish |
| TD1-L1/L2/L3 | LOW | n/a | partial | L2 closed (synthetic BR removed); L1/L3 deferred |

## TD1-C2 (CRITICAL, conf 98) — NOT FIXED. Silent miscompile reconfirmed.

**Location**: `helixc/frontend/typecheck.py:6417-6419` (the `_reject_non_i32_result_payload` helper **explicitly whitelists** `TyResult`):

```python
# Accept nested Result (identity-recurses to i64 packed).
if isinstance(stripped, TyResult):
    return
```

The comment is wrong. Nested Result does NOT identity-recurse — it packs the outer `Ok(inner)` via `RESULT_PACK(0_tag, inner_i64)` at `lower_ast.py:2058-2077`, and the backend at `x86_64.py:2200-2210` reads only the LOW 32 bits of the payload operand via `mov ecx, [rbp+payload_slot]`. The inner Result's tag (which lives at bit 32 of the inner packed i64) is silently destroyed.

**Live repro** (HEAD `b4c8434`):

```python
src = """
fn foo() -> Result<Result<i32, i32>, i32> { Ok(Err(99)) }
fn main() -> i32 {
    let r = foo();
    let inner = unwrap_ok(r);
    if is_err(inner) { 100 } else { 200 }
}
"""
# typecheck: 0 errors
# IR (foo): v2=result.pack(1_tag, 99); v4=result.pack(0_tag, v2_i64); return(v4)
# backend: ELF 4989 bytes
# runtime: exit code 200 (EXPECTED 100 — inner Err tag was lost)
```

The IR layer shows the defect literally: `v4 = result.pack(v3:i32, v2:i64)` — the payload operand v2 is `TIRScalar('i64')` but the RESULT_PACK opcode reads it as i32. No IR validator catches this; no backend check catches it; the typecheck helper opted in.

## TD4-C3 (CRITICAL, conf 95) — NEW. `map_ok`/`map_err` with Result new_value.

**Location**: `helixc/frontend/typecheck.py:4862-4863` (`map_ok`) + `:4898-4899` (`map_err`). Both call `_reject_non_i32_result_payload(arg_tys[1], ...)` which whitelists TyResult — so `map_ok(r, Err(99))` typechecks-clean with the same downstream miscompile as TD1-C2.

**Live repro**:

```python
src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(1);
    let r2 = map_ok(r, Err(99));
    let inner = unwrap_ok(r2);
    if is_err(inner) { 77 } else { 88 }
}
"""
# runtime: exit 88 (EXPECTED 77 — inner Err tag lost via map_ok SELECT → RESULT_PACK)
```

Gate-3 added the G3-H1 fix to map_ok/map_err using the same helper, propagating the TyResult whitelist defect.

## TD1-H1 (HIGH, conf 92) — NOT FIXED. Misleading backend error persists.

`struct Holder { r: Result<i32, i32> }; let h: Holder = Holder { r: Ok(42) }; unwrap_ok(h.r)` still typecheck-clean → IR-lower-clean → backend `NotImplementedError: x86_64 backend LOAD_ELEM/STORE_ELEM does not yet support i64 array elements ... see audit-stage28-8 cycle 16 C16-1`. The error mentions "array elements" while the user wrote a struct — misleading per gate-1 fix-1 recommendation. Tuple-of-Result and array-of-Result also typecheck-clean (not even reaching codegen for tuple, raises mid-IR).

## Items CONFIRMED CLEAN

- TD1-C1: scalar payload widths (i64, u64, f32, f64, bool, u8, u16, u32, i8, i16) all rejected at `Ok`/`Err` construction with the canonical Stage 49 diagnostic naming Stage 50+. Verified 10/10 types.
- ABI integrity at fn-return: `Result<i32,i32>`-returning fn correctly returns i64 in rax (`return` codegen routes through `_is_64bit_int_type` at `x86_64.py:2824`). Caller side similarly correct at CALL result-receive (`x86_64.py:2663`). Cross-check `let r: i32 = helper()` still typecheck-rejects.
- Wrapper-quintet × Result: `let k: Known<Result<i32,i32>> = into_known(Err(99))` typechecks-clean; into_known is identity-lowered, packed-i64 preserved. Wrapper-strip in helper works (8-level guard at lines 6397-6411).
- TD1-L2 (synthetic BR sentinel): removed — codegen-stage TRAP is implicit terminator (per c530891).

## Recommended close action

**RESET gate cycle.** TD1-C2 + TD4-C3 are CRITICAL silent miscompiles that were INTRODUCED into the audit trail at gate-1, partially addressed at gate-2 with a helper that EXPLICITLY re-opened the defect via the `TyResult` whitelist, and propagated to a fresh entry point at gate-3. Minimum fix: delete lines 6417-6419 (`if isinstance(stripped, TyResult): return`) so nested Result rejects at typecheck. The helper's existing diagnostic ("only i32 payloads work today") is appropriate. Also recommend TD1-H1 fix-1 (narrow typecheck reject for aggregate-of-Result) to convert misleading backend errors into source diagnostics.

**VERDICT: FIXES REQUIRED — gate-4 cycle must be reset.**
