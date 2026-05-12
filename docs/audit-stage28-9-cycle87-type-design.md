# Audit Stage 28.9 cycle 87 — Type design

**Scope:** HEAD `d8e5807` (Stage 28.9 cycle-86 audit follow-up: cycle-85 C85-1 SHL/SHR bound fix).

**Mode:** READ-ONLY. Read/Grep/Glob/Bash only. One Write (this file). No Edit performed.

**Stage 28.9 progress:** 0/5 entering this audit. Parallel Stage 28.10 (COMPLETE) and 28.11 INCREMENT 1 commits are independent and were NOT audited.

**Pass criterion:** 0 findings at confidence ≥ 75% = PASS.

---

## Cycle-86 type-surface verification — `_INT_BITS.get(name, 64)` fallback

`helixc/ir/passes/const_fold.py:498-502, 512-516` introduces the SHL/SHR bound check:

```python
_res_ty = op.results[0].ty if op.results else None
_bits = _INT_BITS.get(
    _res_ty.name if isinstance(_res_ty, tir.TIRScalar) else "",
    64,
)
if r < 0 or r >= _bits: raise ShiftFoldError(...)
```

Contrast with `_wrap_int_to_type` at line 121-123: `bits = 32` default; `bits = _INT_BITS.get(ty.name, 32)`.

### Reachability check

`_INT_BITS` covers every Phase-0 integer scalar (i8/u8, i16/u16, i32/u32, isize/usize, i64/u64, plus the bool→32 reified-comparison entry). The SHL/SHR `_res_ty` arrives from `op.results[0].ty`, which the typechecker constrains to an integer TIRScalar for SHL/SHR (floats and tensors are rejected at the typecheck stage, not the const-fold stage). So:

- **bool**: explicit `_INT_BITS["bool"] = 32`. Bound check uses 32. SHL/SHR on bool is itself a typechecker error in Phase-0, so unreachable, but if reached the value is correct.
- **f32 / TIRTensorTy**: blocked upstream — `op.kind in (SHL, SHR)` with float/tensor `res.ty` never enters `_try_fold_op` for Phase-0 modules. The `isinstance(_res_ty, tir.TIRScalar)` guard would return `""` for tensor and hit the 64 fallback, but no SHL/SHR op ever ships with tensor `res.ty`.
- **Future i128 / u128 / unknown scalar**: the 64 fallback would permit shifts up to 63 then `_wrap_int_to_type` at line 535 would narrow to 32 bits (its own fallback). That is the SAME defect-class the cycle-86 fix closed for the known types. But: (1) no such type exists in Phase-0, (2) adding one is a deliberate type-system extension that would have to update `_INT_BITS` as part of the change, and (3) the typechecker is the gatekeeper, not const-fold.

### Intent-consistency observation (NOT a finding ≥ 75%)

The two fallbacks (32 in `_wrap_int_to_type`, 64 in SHL/SHR bound) have asymmetric defensive intent — narrowest-wrap vs most-permissive-bound — but the asymmetry is unreachable in Phase-0 and is dominated by typecheck gatekeeping. Below 75% threshold; documentation-grade nit at most. Deferred.

---

## Rotate-fresh: `helixc/frontend/totality.py` post-ASTVisitor migration — arm completeness over Pattern subclasses

Cycle-71 migrated totality from hand-rolled `_children` / `collect_items` walkers to `iter_fn_decls` + `_SelfCallCollector(ASTVisitor)`. Cycle-73 corrected the double-descent in `visit_Call`.

`_SelfCallCollector` overrides only `visit_Call`. All other node types — including every Pattern subclass (`PatLit`, `PatBind`, `PatWildcard`, `PatTuple`, `PatOr`, `PatRange`, `PatVariant`) — fall through to `ASTVisitor.visit` → `generic_visit`, which uses `dataclasses.fields` introspection in `_iter_child_nodes` to traverse every dataclass-field child node. So:

- `Match` → `MatchArm.pattern` / `.guard` / `.body` all descended generically.
- `PatVariant.sub_patterns` (list of Pattern) — each element descended.
- `PatOr.alts` / `PatTuple.elems` — each element descended.
- `PatRange.lo` / `.hi` (Expr) — descended (catches any Call hiding in a range bound).
- `PatLit.value` (Expr) — descended.

A future Pattern subclass with new dataclass fields (e.g. `PatStruct { path, fields: list[(str, Pattern)] }`, `PatSlice { elems: list[Pattern] }`) is auto-traversed via `dataclasses.fields` — no hand-rolled enumeration to drift. The cycle-71 migration's explicit goal ("drift-proof Pattern-subclass coverage") holds.

No finding.

---

## Findings ≥ 75%

**0.**

## Verdict: **PASS**

Cycle-86 SHL/SHR bound fix is correct and complete for all reachable Phase-0 types; the 32-vs-64 fallback asymmetry is unreachable. Totality.py's post-ASTVisitor migration covers all current and future Pattern subclasses via `dataclasses.fields` introspection — no arm-completeness gap. Stage 28.9 advances 1/5 → 1/5 (no new defects to fix); remaining cycles continue.
