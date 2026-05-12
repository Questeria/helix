# Audit Stage 28.9 cycle 85 — Type design

Scope: HEAD `fb80a4f`

## Rotation

- `helixc/frontend/strings_io.py` — string lowering type invariants
- `helixc/ir/passes/effect_check.py` — effect-set algebra (cycle-16 area, fresh)
- `helixc/ir/passes/cse.py` — op-equivalence type guards

Prior C1–C84 + deferred-known NOT re-flagged per scope.

## Method

Type-design review only. Examined: declared collection types (set vs frozenset
for shared constants), generic parameterization on dict/list/tuple, exception
class shape (trap-id constants, severity Literal), None-handling at API
boundaries, hash-key composition guarding semantic equivalence
(operand ids + attrs primitive/complex split + result-type repr), set-algebra
operations on `frozenset[str]` effect labels, attribute-key classification
discriminators (META_ATTRS bare keys + META_ATTR_PREFIXES value-carrying form),
disjoint runtime check on `_HARD_EFFECT_TRAP_IDS` vs `_INFO_EFFECT_TRAP_IDS`
(post-cycle-32 raises RuntimeError, survives `-O`).

## Scope deviation

`helixc/frontend/strings_io.py` is not present in the tree at HEAD `fb80a4f`.
No string-lowering module exists under either `helixc/frontend/` or
`helixc/ir/`. The rotation entry is reported as scope-vacuous rather than
substituted; no finding is fabricated for an absent file. The two remaining
rotation files were audited in full.

## Findings

None at confidence >= 75%.

`effect_check.py` set-algebra is type-clean (frozenset[str] throughout;
`EffectSeverity` Literal narrows the return type; `eff_errs: list[str] | None`
defensively rejects None; META_ATTRS / META_ATTR_PREFIXES split cleanly
discriminates bare vs value-carrying attribute keys; OP_EFFECTS keyed by
`tir.OpKind` with frozenset values).

`cse.py` `_op_hash` composes (kind, operand_ids, primitive-attrs sorted,
complex-attrs repr sorted, result-type repr) — the result_ty key guards the
audit-10 bool-vs-i32 collision; CAST/BITCAST distinct-target invariant holds.
`seen: dict[tuple, list[tir.Value]]` and `rewrites: dict[int, tir.Value]` are
well-typed; the cycle-18 C18-C1 shallow-copy on `seen[key] = list(op.results)`
preserves the list-identity invariant against later in-place mutation.

## Result

PASS. 0 findings at confidence >= 75%.

No edits made (strict read-only audit; single Write to this doc only).
