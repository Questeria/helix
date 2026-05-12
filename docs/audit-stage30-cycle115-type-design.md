# Stage 30 Cycle 115 Type-Design Audit

Date: 2026-05-12
Verdict: PASS
Confidence: HIGH

## Scope

Checked that Helix integer operations now follow a coherent source-width model across runtime and optimization:

- signed sources sign-extend from their declared width
- unsigned sources zero-extend from their declared width
- unsigned compare/div/mod uses the backend's effective operation width, 32-bit unless a 64-bit operand/result is involved
- casts preserve declared source meaning before destination storage/conversion

## Resolved Design Issues

- `0_u64 + (0_i32 - 1_i32)` and `0_u64 | (0_i32 - 1_i32)` now match `(0_i32 - 1_i32) as u64`.
- Narrow unsigned compare cases such as `u8`/`u16` wrapped values no longer compare using stale raw 32-bit slot contents.
- Mixed narrow signed/unsigned comparisons such as `(0_i8 - 1_i8) == 65535_u16` fold at the same 32-bit effective width as runtime.
- Narrow unsigned and signed casts to integer and float destinations now respect declared source width.
- Signed narrow div/mod and signed i32 guarded div/mod share the same source-width and guard semantics as const-fold.

## Verification

- Focused C115 const-fold tests passed after each fix.
- Focused C115 IR tests passed after each backend encoding change.
- Focused C115 codegen tests passed after each runtime behavior change.
- Final broader checks:
  - 58 IR tests passed
  - 52 const-fold tests passed
  - 32 Stage 30-heavy codegen tests passed
  - 18 bootstrap tests passed

## Stage 30 Status

No blocking type-contract issues remain in Cycle 115. Strict consecutive-clean streak: 1/5.
