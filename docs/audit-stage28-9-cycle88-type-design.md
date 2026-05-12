# Audit Stage 28.9 cycle 88 — Type design

**Scope:** HEAD `e0967670d5c959444ce8d8d09b38e396b7a348ff` (no code change since cycle-86 SHL/SHR bound fix; Stage 28.9 at 1/5).

**Mode:** READ-ONLY. Read/Grep/Glob/Bash only. One Write (this file). No Edit performed.

**Pass criterion:** 0 findings at confidence ≥ 75% = PASS.

**Parallel work:** Stage 28.10 and 28.11 commits NOT audited (independent track).

Prior C1–C87 + deferred items NOT re-flagged.

---

## Rotate-fresh A: `helixc/ir/tir.py` `OpKind` coverage across CSE / DCE / const_fold

Cross-referenced every `OpKind` enum member (lines 125–302) against the three pass dispatch tables: `cse.PURE_KINDS`, `dce.SIDE_EFFECT_KINDS`, and `const_fold._try_fold_op` branches.

Classification:

- **CSE-pure + const-foldable**: CONST_INT/FLOAT, ADD/SUB/MUL/DIV/MOD, NEG, BIT_*, SHL/SHR, BIT_NOT, CMP_*, CAST, MAXIMUM/MINIMUM/POW, BITCAST. Drift between cse (deduped) and const_fold (folded) was the cycle-21 C20-T4 / cycle-22 C22-1 territory and is closed at HEAD.
- **DCE side-effect**: RETURN, BR, COND_BR, CALL, STORE_*, ALLOC_*, MODIFY, SPLICE, PRINT, QUOTE, REFLECT_HASH, ARENA_PUSH/SET, TILE_INDEX_STORE, FFI_CALL, TRAP, TRACE_ENTRY/EXIT. Each annotated with the cycle of audit origin.
- **Pure-but-unoptimized** (not in any list): EXP/LOG/SQRT/RECIP/RELU/GELU/SILU/TANH/SIGMOID/ABS, REDUCE_*, MATMUL/CONV*, RESHAPE/TRANSPOSE/BROADCAST/SLICE/CONCAT, QUANTIZE/DEQUANTIZE, SELECT/WHERE, GRAD/JVP/VMAP, TENSOR_*, STR_BYTE/STR_PTR, THREAD_IDX, TILE_INDEX_LOAD, ARENA_GET/LEN. DCE correctly drops these when their results are unused (they are pure by construction). CSE forfeits dedup on these (conservative miss, not a miscompile). The cycle-21 C20-T4 explanatory comment is explicit that adding LOG/SQRT/EXP/RECIP/ABS is gated on `@safe` semantics — the wider set is the same class. No regression.

### Reachability spot-check

Spread phase in `dce.dce_function` (lines 114–125): liveness propagates backward through pure ops via "result-live → operands-live". Side-effect-op results are not seeded directly, but downstream pure consumers of side-effect-op results still pull them in transitively when the pure result is itself live (via RETURN-seed or block-param). Verified by walking `%y = ADD(%call_result, %c); RETURN(%y)`: %y seeded → %call_result + %c live → ADD kept → CALL kept (side effect, always kept). Algorithm sound.

No finding.

---

## Rotate-fresh B: `helixc/frontend/typecheck.py` `_widen_canon_name` / `_WIDEN_RANK` internal consistency

Checked the cycle-3 C3-2 pointer-width-alias canonicalization (lines 225–232) against the rank table (lines 235–254). `_WIDEN_RANK` assigns `i64 = 40`, `isize = 40`, `u64 = 41`, `usize = 41` — equal-rank pairs for canonical-alias pairs. `_widen_diff_inner` (line 280–292) canonicalizes names before deciding whether to fire the tie-warn callback, so `D<i64> + D<isize>` does NOT emit AD002 (correct: same machine width, no precision/signedness domain to lose). The integer literal-range table (lines 1811–1817) gives isize/usize the same range as i64/u64 — consistent with backend `_is_i64_type` accepting both (x86_64.py:1011).

No finding within audit scope (pre-flatten typecheck deferred).

---

## Rotate-fresh C: `helixc/backend/x86_64.py` register-allocator type invariants

`FnCompiler._alloc_slot` (line 897–901) allocates a uniform 8-byte slot per SSA value, regardless of scalar width. The pre-allocate loop at lines 906–937 covers (1) ALLOC_ARRAY ops, (2) ALLOC_VAR ops, (3) all block params + op results across every block (not just entry), (4) fn params. `_check_float_supported` (line 1019–1028) hard-errors on f16/bf16 before slot allocation — narrow + loud, matches the cycle-3 pattern. `_is_i64_type` / `_is_u64_type` (line 1005–1017) include isize/usize aliases — cycle-19 C18-1 fix holds, no silent 32-bit truncation in spill/load.

Tensor-typed Values (TIRTensorTy / TIRTileTy) are NEVER materialized by the scalar codegen path (verified: zero references to those names in x86_64.py). The 8-byte uniform slot is correct for every scalar type that actually reaches this backend.

The pre-allocation order — arrays first, then vars, then SSA values, then fn-param fallback at line 934–937 (only allocate if not already in `self.slots`) — leaves no path where a Value is referenced via `_slot_of` before allocation. The cycle-init `_op_index` map (line 835–840) is purely for symbol-name determinism (Stage 28.8.1) and doesn't affect type invariants.

No finding.

---

## Findings ≥ 75%

**0.**

## Verdict: **PASS**

Cross-pass OpKind coverage is consistent at HEAD: every kind is correctly classified as pure-foldable, pure-unoptimized, or side-effecting, with prior cycles (C20-T4, C22-1, C13-1, C18-1) having closed the historical drift cases. Typecheck pointer-width canonicalization is internally consistent across the rank table, widening callbacks, and literal-range table. The x86_64 register allocator's uniform 8-byte slot invariant holds for every scalar type that reaches the backend, with isize/usize correctly classified post cycle-19 C18-1. Stage 28.9 progress remains at 1/5 (no new defects).

**No edits performed.**
