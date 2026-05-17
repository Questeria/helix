# Stage 36 post-Inc-14 — Type-Design Audit

## Summary
4 findings (1 MEDIUM, 3 LOW). No CRITICAL or HIGH. The Inc 12-14 work
holds together: register_derivation3 + parent_at are strict-i32,
effect/AD classifications are family-consistent with the Inc 5/9
ancestors, and stdlib `@pure` annotations match their primitives. The
notable real bug is a strictness *asymmetry* between the new and old
provenance accessors (M1). The rest are architectural smells flagged
for Phase 1+.

## Findings

### M1 — Inc 11 C1 strictness asymmetry between new and old parent accessors (MEDIUM, conf 80)

- `helixc/frontend/typecheck.py:2951-2959` (`parent_left_at` / `parent_right_at`)
- `helixc/frontend/typecheck.py:2985-2994` (`parent_at` — new)

`register_derivation` (Inc 11 C1 LOW fix) and `register_derivation3`
(Inc 14, this round) both enforce strict `TyPrim("i32")` on every arg
to avoid the silent i64/u32/u64-truncation hazard during downstream
arena push ops. The new `parent_at` correctly continues that discipline
on both `handle` and `slot`. But the *legacy* `parent_left_at` /
`parent_right_at` checks still use the loose `_is_int_scalar(...)` —
which accepts i64/u32/u64. So after Inc 14 the family now has three
strict members and two loose members:

| Builtin                | handle/idx check     |
|------------------------|----------------------|
| register_derivation    | strict i32 (Inc 11)  |
| register_derivation3   | strict i32 (Inc 14)  |
| parent_at              | strict i32 (Inc 14)  |
| parent_left_at         | `_is_int_scalar`     |
| parent_right_at        | `_is_int_scalar`     |

**Impact**: a Helix caller that stores a `register_derivation` handle
in an i64 (e.g., interop with a Phase-1 hashmap value column) gets
silently truncated by `parent_left_at(h_i64)` — exactly the bug class
Inc 11 C1 was created to prevent. The accessor side was missed because
the Inc 11 C1 fix was framed as "register" side only.

**Fix**: tighten `parent_left_at` / `parent_right_at` to strict i32 in
the same idiom as the new `parent_at` check; reuse the Inc 11 C1
diagnostic phrasing for consistency. ~6 lines, no semantic risk (i32 is
the documented contract; this only rejects programs that were already
truncating silently).

### L1 — Phase-0 arity-tracking limitation is documented in code but not trapped (LOW, conf 90)

- `helixc/frontend/typecheck.py:2978-2984` (typecheck comment)
- `helixc/ir/lower_ast.py:2150-2163` (lowering comment)

Both code sites already document the Phase-0 limitation: `parent_at(h,
2)` on a two-parent handle reads into "whatever happens to live at slot
N+2, which may be another derivation's slot or the OOB sentinel". This
is correct disclosure — the doc is in the code, not only the progress
ledger.

What's missing: no `TODO` / `FIXME` marker grep-able for the
Phase-1 cleanup, and no runtime trap. The current contract is "user is
responsible for slot < arity", which is a documentation-only invariant
— the type-design anti-pattern this skill flags. Phase-0 acceptable
(arity in the handle would need an ABI change to make `register_derivation*`
return a tagged struct, which we don't want to retrofit now), but the
risk surface grows once a 4-parent variant lands.

**Fix (deferred to Phase 1)**: pack arity into the high bits of the
i32 handle (e.g., top 4 bits = arity 0-15, low 28 bits = slot) so
`parent_at` can bounds-check `slot < arity_of(handle)` and return -1
on violation. Add a `# TODO(stage37-arity-in-handle):` marker at both
sites *now* so the cleanup is grep-able.

### L2 — parent_at and parent_left_at lower via independent code paths (LOW, conf 75)

- `helixc/ir/lower_ast.py:2104-2140` (parent_left_at: `SUB(h,1)` then `_safe_arena_get(base, 1)`)
- `helixc/ir/lower_ast.py:2141-2167` (parent_at: `SUB(h,1)` then `ADD(base, slot)` then `_safe_arena_get(eff, 0)`)

The Inc 14 back-compat invariant `parent_at(h, 0) == parent_left_at(h)`
and `parent_at(h, 1) == parent_right_at(h)` is currently pinned by ONE
test (`test_stage36_inc14_parent_at_on_two_parent_handle_back_compat`).
The two lowerings are independent: parent_left_at bakes the +1 offset
into the `_safe_arena_get` displacement; parent_at computes the offset
dynamically via ADD. Both end up calling `_safe_arena_get`, so the
bounds-check sentinel behaviour does match, but a future change to
either path (e.g., a peephole pass that folds `_safe_arena_get(base,
const)` differently from `_safe_arena_get(eff, 0)`) could silently
diverge them.

**Impact**: low — one regression test catches the runtime divergence,
and the IR structure is small enough that drift is unlikely in
Phase-0. Architectural smell, not a bug.

**Fix (deferred)**: collapse `parent_left_at(h)` and `parent_right_at(h)`
to thin wrappers that emit the *same* IR sequence as `parent_at(h, 0)`
/ `parent_at(h, 1)` (with `slot` as a `const_int`). Single source of
truth for the offset arithmetic. Defer until Inc 16 or whenever the
next parent-accessor lands — premature consolidation now would
churn the Inc 9 A1 bounds-check call site.

### L3 — ARENA_PUSH_TRIPLE is a structural clone of ARENA_PUSH_PAIR (LOW, conf 70)

- `helixc/ir/tir.py:254-263` (PAIR + TRIPLE opcode pair)
- `helixc/backend/x86_64.py:2685-2775` (two near-identical codegen blocks ~47 lines each)
- `helixc/ir/passes/dce.py:56-62`, `effect_check.py:93-99` (matching duplicate registrations)

The Inc 14 codegen for ARENA_PUSH_TRIPLE is a copy-paste of
ARENA_PUSH_PAIR with one extra slot. Both opcodes carry the same
arena-effect, the same overflow → -1 contract, the same fail-closed
"none-or-all" atomicity. The duplication is small now (one extra
opcode, ~47 lines of codegen), but the pattern doesn't scale: a
hypothetical 4-parent or N-parent variant would either keep cloning or
require a backfit refactor.

**Impact**: zero in Phase-0 — both opcodes work correctly and the test
suite pins their contracts. Architectural smell for Phase 1+.

**Fix (deferred to Inc 15+ or Phase 1)**: replace both with a single
parametric `ARENA_PUSH_N` whose arity is encoded as a constant operand
(or an immediate on the opcode). Codegen becomes a small loop over the
operand list; DCE/effect-check tables collapse. Worth doing BEFORE the
next variant (Inc 15? Inc 16?) lands so we don't pay the migration cost
twice. If only 2-parent + 3-parent exist long-term, leave the clone.

---

## Cross-cuts checked & clean

- **Effect/purity classification of new builtins**: `parent_at` added
  to `AD_KNOWN_PURE_CALLS` (autodiff.py:85), `register_derivation3`
  deliberately NOT added — symmetric with the Inc 9 B2 family
  discipline (write-side is impure, read-side is pure). The
  NotImplementedError-on-AD path for `register_derivation3` is
  inherited correctly from the opaque-call default.
- **ARENA_PUSH_TRIPLE registered in both DCE side-effect set and
  effect_check.OP_EFFECTS["arena"]**: yes, both sites updated
  consistently (dce.py:59, effect_check.py:97).
- **stdlib `@pure` consistency** (provenance.hx): `has_evidence` and
  `evidence_left/right` are `@pure` and only call pure primitives
  (`parent_left_at` / `parent_right_at`). `trace_evidence` is correctly
  NOT `@pure` (it calls `print_str` / `print_int`). All correct.
- **1-based handle invariant** (Inc 9 A2): `register_derivation3`
  lowering adds the `+ 1` via the same `ADD push_idx, one` shape used
  by `register_derivation`. Overflow → push_idx = -1 → handle = 0 = null
  sentinel — fail-closed contract preserved.
- **Integer-Logic AD guard test coverage** (Inc 12 catch-up): 6 tests
  added covering forward + reverse, with-twin + no-twin branches —
  closes the Inc 11 B2 deferral cleanly.
- **Sentinel-as-value (0 = null, -1 = OOB)**: known Phase-0 design;
  acceptable until Phase-1 Result<T, ProvErr> lands. No new
  pollution from Inc 12-14 — both sentinels were already in the i32
  return type from Inc 5/9.

## Recommendation

Land M1 as a small follow-up (Inc 14.1 or rolled into Inc 15
opener — 6-line strict-i32 tighten on `parent_left_at` /
`parent_right_at`). L1-L3 are architectural; tag them with grep-able
TODOs now, defer the actual rework to Phase 1 or to whenever a
4-parent variant forces ARENA_PUSH_N anyway. Inc 14 itself is sound.
