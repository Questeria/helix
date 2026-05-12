# Stage 28.8 pre-29 audit gate — Cycle 25 (Audit B: type-design soundness)

**Date:** 2026-05-11
**HEAD:** `6db467f` ("Audit 28.8 cycle 23+: close C22-C (HIGH, match_lower walker drift)")
**Lens:** type-design soundness (Audit B)
**Streak counter at start:** 2/5 (cycles 23, 24 both clean)
**Bar:** ZERO findings of ANY severity at confidence >= 75. Re-flagging
prior-cycle findings is forbidden.

---

## Scope — stability re-pass + new-surface verification

Two code commits exist between cycle-24 HEAD `4bdc800` and cycle-25 HEAD
`6db467f`:

| Commit | Surface | Audit-trail provenance |
|---|---|---|
| `89d49e9` | `helixc/ir/passes/effect_check.py` (+34 lines) | C22-1/2/3/4/5 fix-sweep |
| `6db467f` | `helixc/frontend/match_lower.py` (+34 lines) | C22-C walker drift fix |

Both commits close cycle-22 findings already audited and approved. This
cycle's job: verify the autonomous fixes introduce no new type-design
soundness gap, and re-affirm cycle-22 type-design ledger.

---

## New surface 1: `effect_check.py` OP_EFFECTS extensions

`OP_EFFECTS: dict[tir.OpKind, frozenset[str]]` gained 7 new entries
(FFI_CALL, ARENA_PUSH, ARENA_SET, QUOTE, REFLECT_HASH, TILE_INDEX_STORE,
TRACE_ENTRY, TRACE_EXIT) and 5 new effect-label strings (`ffi`, `arena`,
`reflect`, `tile_io`, `trace`). `callees()` gained one new dispatch arm
for FFI_CALL.

### Type-soundness verification

- The map's value type is unchanged (`frozenset[str]`). New entries
  conform to the existing structural shape.
- The new label strings round-trip cleanly through `declared_effects`
  via the `effect:<name>` namespacing convention (line 125-130).
  A user writing `@effect(ffi)` produces attr key `effect:ffi`, which
  `declared_effects` strips to `"ffi"` — matching the label
  `OP_EFFECTS[FFI_CALL]` contributes to the closure. No 19002 false
  positive, no 19001 escape.
- `callees()`'s new FFI_CALL arm preserves the existing return-type
  contract (`set[str]`) and the `<indirect-ffi>` sentinel mirrors the
  pre-existing `<indirect>` convention. The `compute_closure` fixpoint
  treats both sentinels identically (line 193-194 maps `<indirect>` to
  `"unknown"`; an `<indirect-ffi>` name not in `module.functions` falls
  through to line 198-199, also contributing `"unknown"`). Soundness
  preserved: indirect FFI conservatively pollutes the closure with
  `unknown`, never silently drops the effect.
- No new dataclass fields, no new generic parameters, no new dispatch
  enums. The change is purely additive entries in an existing
  enum-keyed mapping.

## New surface 2: `match_lower._rewrite_expr` dispatch-arm extensions

Six new isinstance arms (UnsafeBlock, Range, Modify, Break, Quote,
Splice). All call existing `_rewrite_expr` (returns `A.Expr`) or
`_rewrite_block` (mutates `A.Block` in-place) and reassign the result
back into existing AST fields whose declared types already accept
`A.Expr`. No new signature, no new type, no narrowing of existing
return types. The walker's surface grows monotonically; type contracts
are preserved.

This is a walker-completeness fix, not a type-design surface change.
It belongs to Audit C (silent failures) under the cycle taxonomy. The
type-design surface for this commit is the empty set.

---

## Cross-target regression ledger (cycle 22 onwards)

| Target | Touched by 89d49e9 / 6db467f? | Status |
|---|---|---|
| `ast_walker.py` field-introspection | No | CLEAN preserved |
| `_op_suffix` collision | No | CLEAN preserved |
| isize/usize cross-pass (13 sites) | No | CLEAN preserved |
| Deferred grad_pass rewriter | No | CLEAN preserved |
| `struct_mono.py` C-fix delta | No | CLEAN preserved |
| `effect_check.py` OP_EFFECTS additions | Yes | additive, type-sound |
| `match_lower.py` walker arms | Yes | walker-only, no type surface |

No finding at any severity meets the 75-confidence bar.

---

## Streak verdict

Cycle 25, Audit B (type-design): **CLEAN** under the strict criterion.

Streak advance:
- Cycle 23: 1/5
- Cycle 24: 2/5
- Cycle 25 (B clean — pending A): **3/5 if A also clean**, else holds
  at 2/5.
