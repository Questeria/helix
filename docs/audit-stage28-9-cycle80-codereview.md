# Audit Stage 28.9 cycle 80 — Code review

Scope: HEAD `d218e65` (cycle-79 fix-sweep: FFI float-return + 2 regression tests).
Reviewer: read-only static review of `helixc/backend/x86_64.py` lines 1779–1795 and
`helixc/tests/test_ffi.py` + `helixc/tests/test_ir.py` regression cases.
Verification: empirical ELF emission probes (no source edits).

## Verdict

**FAIL.** 2 findings at confidence >= 75%.

## Findings

### C80-1 (HIGH conf 90) — `test_c76_1_ffi_call_routes_f32_args_to_xmm0` byte-pattern is non-discriminating

The test asserts `b"\xf3\x0f\x10" in elf` and `b"\xf3\x0f\x11" in elf` to prove the FFI
arg-load and return-store use `movss xmm0, [rbp-N]` / `movss [rbp-N], xmm0`. Empirical
probe shows both byte sequences are emitted by ordinary intra-Helix f32 parameter
prologue spills with no FFI call site at all:

    src = 'fn helper(x: f32) -> f32 { x } fn main() -> i32 { 0 }'
    -> b'\xf3\x0f\x10' in elf  # True
    -> b'\xf3\x0f\x11' in elf  # True

Because the test source defines `fn entry(x: f32) -> f32`, the `movss`-load and
`movss`-store byte sequences appear in `entry`'s own prologue/epilogue regardless of
how the FFI call site lowers the `sinf` invocation. A regression that re-routed
FFI_CALL float args back through INT_REGS (the C76-1 defect) would still emit `movss`
for the `entry` parameter, so the assertion would still pass and the test would NOT
fail. The test name + docstring claim "FFI f32 arg load to xmm0" / "FFI f32 return
store from xmm0", but the assertion has no FFI-call-site anchoring (no surrounding
`call qword [rip+disp]` opcode, no slot-offset match, no relative byte position).
Confidence high: probe is deterministic and reproducible.

### C80-2 (HIGH conf 85) — `test_c76_f1_for_range_i64_increment_dtype_matches_iterator` allows pass via user literal `1_i64` in body

The test source contains `total += 1_i64;` inside the loop body. That user literal
lowers to `CONST_INT(value=1, ty=i64)` independently of the for-range increment
under test. The assertion scans every block for `CONST_INT(value=1, ty=i64)` and
succeeds if ANY exists. Empirical probe: with the body changed to `total += 1;`
(untyped, which would expose the for-range bug), the emitted IR contains BOTH a
`(1, i32)` (the bug-shaped body increment) AND a `(1, i64)` (the for-range
increment). A regression on ONLY the for-range increment (back to i32) would still
leave the body's `1_i64` user literal present, so `increment_ones` is non-empty and
the assertion passes. The docstring claims the test targets the for-range loop's
increment specifically; the assertion structure cannot distinguish iterator-
increment from body literal. Fix would be to delete the `total += 1_i64` line (or
replace with a non-`1` value) and/or filter `CONST_INT` ops to those reachable from
the for-range increment block. Confidence high: IR enumeration is deterministic.

## Notes

- Cycle-79 FFI float-return code change itself (lines 1779–1795) is correct:
  `_is_f64_type` checked before `_is_float_type` so f32 routes correctly; f16/bf16
  are rejected upstream by `_check_float_supported`. Comment cites C78-1 and the
  cycle-77 arg-side counterpart accurately. No drift between comment and code.
- C1–C79 + deferred-known: NOT re-flagged per scope.

No edits performed (strict read-only mode).
