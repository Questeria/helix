# Audit Stage 28.9 cycle 92 — Silent failures

**Scope:** HEAD `d04e65b`
**Mode:** read-only audit
**Threshold:** finding requires conf >= 75%

## Cycle-91 verification

`grep "Stage 28.9 cycle 91 audit-CR C90-1"` returns **4 matches** in
`helixc/tests/test_codegen.py` (the four `*_legacy_api` docstrings).
Cycle-91 docstring-clarification fix confirmed landed.

## Rotation surfaces scanned

1. `helixc/ir/passes/effect_check.py` — effect-set propagation edges
   (declared_effects bare-name fallback, _is_meta_attr value-prefix
   filter, compute_closure indirect-ffi sentinel, PURITY_OBSERVER
   subtract symmetry on @pure vs non-@pure).
2. `helixc/ir/passes/cse.py` — CAST/CMP op-equivalence hashing under
   mixed signedness (signedness via TIRType repr in `_op_hash`
   attrs_complex; result_ty_key disambiguator; PURE_KINDS includes
   bitwise + shift ops post-cycle-21 C20-T4 hardening).
3. `helixc/backend/ptx.py` — silent fallthrough at line 332
   (`// TODO: <kind>`) for unhandled TileOpKind cases.

## Findings (>= 75% confidence)

**None.**

### Sub-threshold observations (NOT findings, recorded for trail)

- ptx.py line 332 silent fallthrough is documented Phase-0 stub
  behavior ("This is a STUB — real codegen ... lands incrementally").
  Of the unhandled TileOpKind members, only SCALAR_SELECT is in
  `TirToTileLowerer.SCALAR_OP_MAP` (tile_ir.py:170), but
  `lower_ast.py` never emits `tir.OpKind.SELECT` (grep negative;
  only test_select_codegen.py hand-builds it, and that test targets
  x86 not PTX). Tensor tile ops (TILE_LOAD_GLOBAL, TMA_*, TILE_ADD,
  TILE_MATMUL, etc.) are not in SCALAR_OP_MAP and the lowerer emits
  TODO markers per its own documented Phase-0 policy
  (tile_ir.py:154). Reachability via normal compilation = 0.
  Confidence as a silent-failure finding: ~50%. Below threshold.
- effect_check.declared_effects' `v is not True` guard correctly
  rejects value-carrying payloads; META_ATTR_PREFIXES covers
  parser's known value-attr shapes. No edge case found.
- cse._op_hash includes `result_ty_key` + `attrs_complex` (repr of
  TIRType objects in from_ty/to_ty). Helix currently has no unsigned
  integer type (tir.py:155 — "Helix has no unsigned int type yet"),
  so the "mixed signedness" surface is empty in Phase 0. CAST hashing
  distinguishes by from_ty/to_ty repr; CMP ops have kind-distinct
  enum members (no kind merging risk).

## Verdict

**PASS** — 0 findings at conf >= 75%.

## Edits

No edits performed. Audit is read-only. One Write to this doc only.
