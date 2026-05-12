# Audit Stage 28.9 cycle 99 — Type design

Scope: HEAD 32c66bf99b3378a7f67ccd9854794fbc9a4e2bf9

Files under fresh-rotation review:
- helixc/ir/passes/effect_check.py — effect-set algebra
- helixc/ir/passes/cse.py — hash-key construction
- helixc/backend/ptx.py — PTX type-to-machine mapping

Prior cycles C1-C98 + all deferred items NOT re-flagged.
Stage 28.10 / 28.11 parallel audits are INDEPENDENT and out of scope.

## Method

Read-only static review. Verify type-level consistency of:

1. effect_check.py — frozenset[str] algebra across OP_EFFECTS,
   PURITY_OBSERVER_EFFECTS, META_ATTRS, declared_effects, own_op_effects,
   compute_closure, check_module; fixpoint termination; trap-id
   classifier disjointness; EffectSeverity Literal coverage.
2. cse.py — `_op_hash` tuple-key shape: (OpKind, tuple[int], tuple[(str,prim)],
   tuple[(str,str)], str|None); hashability / dict-key safety; result-ty
   discrimination for single-result pure ops; block-scoped seen-dict soundness.
3. ptx.py — `_ptx_type_str`, `_DTYPE_SIZE`, `_DTYPE_PTX_LOAD`,
   `_ld_reg_prefix` mapping coverage; isize/usize/bool entries; per-prefix
   register pool integrity; param-space b64 vs scalar-width mismatch
   surface.

## Findings (confidence >= 75%)

None.

### Rationale

- **effect_check.py algebra**: All set operations (`clos -
  PURITY_OBSERVER_EFFECTS`, `(clos - PURITY_OBSERVER_EFFECTS) - decl`,
  `decl - clos - {"unknown"}`) operate on frozenset[str] inputs and
  return frozenset[str] / set[str] uniformly. The closure fixpoint
  terminates: closure[n] only grows monotonically over the finite
  universe of effect labels (union of OP_EFFECTS values + {"unknown",
  "ffi"}), and the outer loop's `changed` flag is set only on growth.
  Indirect/FFI sentinels (`<indirect>`, `<indirect-ffi>`) are
  separately handled before the `c in module.functions` branch, so an
  unknown name cannot shadow an indirect sentinel. The cycle-32 C31-4
  runtime disjoint check on `_HARD_EFFECT_TRAP_IDS` vs
  `_INFO_EFFECT_TRAP_IDS` is `-O`-safe (raise, not assert).
  `EffectSeverity = Literal["hard","info","unknown"]` exhaustively
  covers `classify_effect_error`'s return paths.

- **cse.py `_op_hash`**: The 5-tuple key `(OpKind, tuple[int],
  tuple[(str, int|float|str|bool)], tuple[(str, str)], str | None)`
  is fully hashable: OpKind is enum, all interior leaves are hashable
  primitives (or repr'd to strings for complex attrs / result types).
  result_ty_key uses `op.results[0]` which is sound because every op
  in PURE_KINDS produces exactly one SSA result; the audit-10 bool/i32
  MUL discrimination fix is preserved. Block-scoped `seen` and
  `rewrites` dicts correctly prevent cross-block CSE rewrites (no
  dominance analysis available). The cycle-18 C18-C1 defensive copy
  of `op.results` into `seen[key]` neutralizes aliasing risk.

- **ptx.py type mapping**: `_ptx_type_str` covers i8/i16/i32/i64,
  u8/u16/u32/u64, isize/usize (both 64-bit per cycle 21 C20-1),
  bool→.pred, f16/bf16/f32/f64. `_DTYPE_SIZE` and `_DTYPE_PTX_LOAD`
  mirror each other with the same key set (now including bool→1/u8
  per cycle 35). `_ld_reg_prefix` returns "f" for float family, "rd"
  for i64/u64/isize/usize, "r" otherwise — consistent with the kernel
  prolog's `.reg .pred %p<256>; .reg .b32 %r<256>; .reg .b64 %rd<256>;
  .reg .f32 %f<256>` declarations. Param-space `.b64 param_<idx>` is
  uniformly emitted regardless of scalar element type, which is
  correct because kernel arguments are always pointer-sized on the
  param-space side; the scalar element type only matters at the
  load-suffix level (`ld.global.<suffix>`). Per-prefix overflow
  detection in `_new_reg` raises RuntimeError with a clear codegen-
  site message.

## Verdict

PASS — 0 findings at conf >= 75%.

## Process note

No edits were made to source files. Only this one audit document was
written. Strict read-only mode preserved.
