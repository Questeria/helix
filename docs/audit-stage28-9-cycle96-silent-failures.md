# Audit Stage 28.9 cycle 96 — Silent failures

Scope: HEAD `56fa3df` (cycle-95 fix-sweep).

Files audited (fresh-rotation): `helixc/ir/passes/cse.py`,
`helixc/ir/passes/effect_check.py`, `helixc/backend/x86_64.py`. Plus
cycle-95-fix verification: `helixc/frontend/typecheck.py:387-394` vs
`helixc/frontend/lexer.py:338-341`, and parser split-based suffix-strip
audit (`helixc/frontend/parser.py`).

Mode: STRICT READ-ONLY. No edits made.

## Verdict: FAIL — 1 finding at confidence >= 75%

## Cycle-95 fix verification

- `_FLOAT_PRIM_NAMES` (`typecheck.py:387-390`) = `{f16, bf16, f32, f64,
  fp8, mxfp4, nvfp4, ternary}` and lexer suffix whitelist
  (`lexer.py:338-341`) = same 8 names + 10 int names. The two are now
  in sync at the frontend layer for the kind-coherence check.
- `Parser._parse_autotune_int` (`parser.py:360-374`) uses `t.int_value`
  directly. Scanned the parser for other `t.value.split("_")[0]` style
  patterns that could re-introduce the digit-separator bug:
  `IntLit`/`FloatLit` construction at `parser.py:1087-1092` reads
  `t.int_value` / `t.float_value`; no other split-based suffix-strip
  pattern survives. Verified clean.

## Findings

### F1 (HIGH, conf 90) — cycle-95 `_FLOAT_PRIM_NAMES` expansion leaves the x86_64 backend unable to lower `fp8` / `mxfp4` / `nvfp4` / `ternary`; codegen silently treats them as i32

Cycle-95 widened the frontend kind-coherence check to admit the four
quantized-float / low-precision suffixes (`fp8`, `mxfp4`, `nvfp4`,
`ternary`) into `_FLOAT_PRIM_NAMES`, AND `typecheck.PRIMITIVES`
(`typecheck.py:336-343`) already accepts those names as nominal
types. Net effect after cycle-95: `let x: fp8 = 1.0_fp8;` and
`fn f(x: mxfp4) -> mxfp4 { x }` are now well-typed programs that lower
to `TIRScalar("fp8")` / `TIRScalar("mxfp4")` results and parameters.

The x86_64 backend has not been updated in lockstep:

- `x86_64.py:999-1000` — `_is_float_type` returns True only for
  `{f16, bf16, f32, f64}`. Returns **False** for `fp8`, `mxfp4`,
  `nvfp4`, `ternary`.
- `x86_64.py:1019-1028` — `_check_float_supported` raises
  `NotImplementedError` only for `{f16, bf16}`. Falls through silently
  for `fp8`, `mxfp4`, `nvfp4`, `ternary`.
- `x86_64.py:1005-1011` — `_is_i64_type` returns False (name set
  `{i64, isize}` doesn't include these names either).

Consequence — for a `fp8`-typed param / SSA value / CONST_FLOAT
result, the backend takes the `else` branch in every type-dispatched
op site, including:

- Arg spill in the prologue (`x86_64.py:971-990`) — int-register spill
  (`mov_mem_rbp_edi` etc.) instead of `_movss_store_xmmN`. An ABI-level
  float arg in `xmm0` is silently lost; the int-register slot picks up
  whatever caller-side value happens to be in `edi`.
- `RETURN` (`x86_64.py:1809-1826`) — `mov_eax_mem_rbp` is the
  fall-through branch, so the returned value is sent back as an i32 in
  `eax` rather than as a float in `xmm0`.
- CONST_FLOAT (`x86_64.py:1160-1180`) and the float-arithmetic ops
  (ADD/SUB/MUL/DIV at lines 1267-1356) — they branch on `_is_f64_type`
  then `_is_float_type`; both False for these names, so codegen
  reaches the integer path or fails to emit anything depending on op.

Reproducer shape (verified by reading; not executed in this audit):

```helix
fn f(x: fp8) -> fp8 { x }
fn g() -> fp8 { 1.0_fp8 }
```

Both type-check post cycle-95. Both lower. The backend then mis-uses
the integer ABI for the float-domain value with **zero diagnostic**.
This is the same defect class as the cycle-93 / 94 / 95 frontend
sweep (cross-domain bit reinterpretation) but moved one stage
downstream — typecheck now accepts what the backend cannot represent.

Same defect class as cycle-3 (`narrow + loud`) and cycle-19 C18-1
(`isize` alias miss): a typecheck pass admits a type the backend
classifier silently mis-buckets. The cycle-19 fix was a one-line
`_is_i64_type` widening. The minimum fix here is:

- Extend `_check_float_supported` to also raise on
  `{fp8, mxfp4, nvfp4, ternary}` — there is no SSE/AVX path for these
  on a stock x86_64 target, so "narrow + loud" is the correct
  Phase-0 stance. (`_is_float_type` would still need updating if/when
  a real codegen path is added; but the loud-error blocks the silent
  miscompile today.)

Confidence 90: the code path is direct (frontend admits → IR carries
the name → backend type predicates are name-equal only → all paths
fall through to int defaults), the bug class is identical to two
prior cycles' findings (cycle-3 f16/bf16; cycle-19 isize/usize), and
there is no intermediate guard. Lower than 95 because I did not
execute a runtime reproducer in this read-only audit; the silent
miscompile is inferred from static code reading.

## Other observations (below 75% confidence threshold — not findings)

- `cse.py` `_op_hash` (line 76-91) uses tuple-of-primitives. CONST_FLOAT
  with `-0.0` vs `0.0` would hash-collide (`hash(-0.0) == hash(0.0)`
  in Python and `-0.0 == 0.0` is True), so two CONST_FLOAT ops with
  values `0.0` and `-0.0` would CSE into the first. This matters for
  IEEE semantics (`1.0 / -0.0 == -inf` vs `1.0 / 0.0 == +inf`).
  Confidence low (~50): user-written Helix would rarely have both
  literals in the same block, and CONST_FLOAT is hoisted/folded
  before this stage in many cases. Logged as a future-cycle defense
  candidate, not a current finding.
- `effect_check.OP_EFFECTS` (line 67-110) has no entry for
  `ARENA_GET` / `ARENA_LEN`. By documented policy these are reads, not
  effects, so absence is correct. Logged for completeness.
- `effect_check.callees()` (line 254-279) iterates CALL, FFI_CALL,
  MODIFY (verifier_fn). SPLICE / QUOTE / REFLECT_HASH do not have
  function-target attrs; the existing OP_EFFECTS labels they carry
  (`reflect`) are already propagated via `own_op_effects`. No callee
  miss.

## Re-checks of recently-audited paths (not re-flagged)

Per scope rules, deferred-known items (`monomorphize._mangle_ty`
silent catchall, `hash_cons._ast_equal`, `typecheck/struct_mono`
pre-flatten in `check.py`, `autotune.collect_autotuned_fns
iter_fn_decls`, `struct_mono.mangle_struct` collision) were not
re-examined.

## Summary line

FAIL — 1 finding at conf >= 75%. F1 (HIGH, conf 90): cycle-95
expanded `_FLOAT_PRIM_NAMES` and `PRIMITIVES` admit `fp8` / `mxfp4` /
`nvfp4` / `ternary` as float-domain types, but the x86_64 backend's
`_is_float_type` and `_check_float_supported` predicates (`x86_64.py`
lines 999, 1019) still hard-code `{f16, bf16, f32, f64}` — every
quantized-float value silently falls through to the integer code path
in arg spill, RETURN, arithmetic, and CONST_FLOAT with no
diagnostic. Same defect class as cycle-19 C18-1 (isize alias miss).
