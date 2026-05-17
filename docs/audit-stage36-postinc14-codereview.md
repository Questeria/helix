# Stage 36 post-Inc-14 — Code Review

Scope: commits `abef645` (Inc 12 catch-up), `6894348` (Inc 13), `e7c3552` (Inc 14)
plus HANDOFF refresh `13784a9`. Reviewed against the explicit checklist in the
audit prompt (codegen, atomicity, lowering, test rigor, stdlib helpers, Inc 12
catch-up coverage, HANDOFF facts).

## Summary

CLEAN — 0 HIGH/CRITICAL findings. 3 MEDIUM-confidence observations recorded
below as L1..L3 (test-coverage / documentation tightening). None block.

## Verification highlights (what I checked and what looks right)

**x86_64 ARENA_PUSH_TRIPLE codegen** (`helixc/backend/x86_64.py:2729-2774`):

- Operand materialization order is correct. left → edx, middle → r8d
  (REX.R + 89 C0), right → r9d (REX.R + 89 C1). edx/r8d/r9d are not aliased
  to anything the surrounding sequence later loads, and they are preserved
  across `lea rax,[rip+__helix_arena_base]` and `mov ecx,[rax]`.
- Bounds check uses `cmp ecx, CAP-2; jb in_bounds`. With slots 1..CAP
  available (slot 0 = cursor) and writes to (cursor+1, cursor+2, cursor+3),
  the maximum acceptable cursor is CAP-3, i.e. cursor < CAP-2. Matches.
- Jump-displacement arithmetic verified byte-for-byte:
  - `jb +7` skips `mov_eax_imm32(-1)` (5 bytes via B8 ..) + `jmp +24` (2 bytes) = 7. ✓
  - `jmp +24` skips writes (4+5+5 = 14) + advance (3+2 = 5) + result recompute (3+2 = 5) = 24. ✓
- Cursor advance is `add ecx, 3` then `mov [rax], ecx`, then `sub ecx, 3` to
  recover the old cursor for the result. The `+3 / -3` symmetry is correct.
  (The alternative of stashing the pre-bump cursor in another reg would save
  3 bytes but require re-allocating a scratch — the chosen layout mirrors
  ARENA_PUSH_PAIR for auditability.)
- Overflow path: `mov eax, -1` lands in eax before `jmp store_result`, and
  `store_result` is the single `mov [rbp+res_slot], eax` at the bottom. The
  -1 sentinel cannot be skipped or overwritten.

**Atomicity**: ARENA_PUSH_TRIPLE issues three unconditional writes after the
bounds check passes. No intervening calls, no conditional store, no early
return between the writes. Cursor only advances on the success path (the
overflow path takes the jmp before `add ecx, 3`). Matches the PAIR pattern
exactly. The Inc 14 atomicity test
(`test_stage36_inc14_arena_push_triple_atomic_against_intervening_push`,
line 2141) actively exercises the contract across an intervening
`__arena_push`.

**Lowering** (`helixc/ir/lower_ast.py:2017-2036, 2152-2167`):

- `register_derivation3`: ARENA_PUSH_TRIPLE returns the old cursor (slot
  index of left, 0-based). Adding 1 converts to a 1-based handle. On
  overflow, push_idx = -1 → handle = 0 = null sentinel, matching the PAIR
  fail-closed contract.
- `parent_at(h, slot)`: SUB-then-ADD is `(h - 1) + slot`. Boundary cases
  verified mentally: h=0 → -1 + slot, passed to `_safe_arena_get` which
  returns -1 when `eff_idx < 0` (signed CMP_GE). h=INT_MAX overflows the
  ADD to negative, also caught by the same check. The dynamic-slot
  treatment is correct — bookkeeping happens at IR not in the displacement.
- `_safe_arena_get` (existing helper, re-used) uses signed CMP_GE / CMP_LT
  + SELECT-clamped speculative ARENA_GET. Reads always land within the
  arena bytes; out-of-range reads return the -1 sentinel.

**Effects + DCE**: `ARENA_PUSH_TRIPLE` is in `OP_EFFECTS` (`{"arena"}`) and
`SIDE_EFFECT_KINDS`. Both are explicitly pinned by Inc 14 unit tests
(`test_stage36_inc14_arena_push_triple_is_in_effect_table`, line 2286;
`test_stage36_inc14_arena_push_triple_is_in_dce_side_effect_set`, line 2297).

**Test rigor (Inc 14)**:

- "42 * has_evidence(h)" / "42 * h" patterns correctly fail-closed against
  regression-to-zero. Tests at lines 2022, 2098, 2116, 2141 all distinguish
  "actual 42" from "regression to 0".
- `test_stage36_inc14_three_parents_all_recoverable` (line 2083) uses
  distinct values (10, 14, 18) summing to 42 — any slot-shuffle would
  break it. Good.
- `test_stage36_inc14_independent_triples_stay_independent` (line 2116)
  weighted sum (1+2+3+4+5+6)*2 = 42 — any aliasing across handles would
  fail.
- `test_stage36_inc14_parent_at_on_two_parent_handle_back_compat` (line
  2171) is exactly the cross-check Inc 14 needs: parent_at(h, 0/1) must
  agree with parent_left_at(h)/parent_right_at(h) on a 2-parent handle.
  Pins the shared-base invariant.
- Typecheck strictness (i32 only) pinned for both `register_derivation3`
  and `parent_at` (lines 2228, 2246).
- Null-handle and OOB-slot fall-through pinned (lines 2194, 2210).

**Inc 13 stdlib** (`helixc/stdlib/provenance.hx`):

- `has_evidence(h)` semantics — `handle <= 0` OR `parent_left_at(h) == -1`
  → returns 0; else 1. The "left==-1" arm only fires when the bounds-check
  sentinel from `_safe_arena_get` triggers (i.e. eff_idx < 0 OR
  eff_idx >= cursor). The edge case I worried about — handle=1 before any
  push (cursor still 0) — is correctly rejected: cursor=0 makes
  `CMP_LT(0, 0)` false, returning -1. Verified by walking the
  `_safe_arena_get` chain at `lower_ast.py:2051`.
- `evidence_left`/`evidence_right` are trivial aliases. Pure.
- `trace_evidence`: side-effecting (calls print_*), correctly NOT marked
  @pure. Returns has_evidence(handle) at the end, so the caller can still
  branch on validity. Stdout assertion in
  `test_stage36_inc13_trace_evidence_stdout_format` (line 1949) is exact.

**Inc 12 catch-up** (`abef645`): the 6 added tests do exercise the guard:
they import `differentiate` / `differentiate_reverse` directly and assert
`NotImplementedError` with regex-matched messages naming the suggested
fuzzy twin (or, for `if_logic`/`to_logic_bool`, the general-guidance
string). The let-erasability negative control (line 1770) confirms the
guard fires only when AD differentiates *through* the call, not when an
unused let with `and_logic` is dropped. Coverage is real.

**HANDOFF_FOR_CLAUDE.md**: the diff covers Inc 1-12 only because Inc 13/14
were committed after the HANDOFF refresh in `13784a9`. As of `13784a9` the
counts ("94 passing", "23+ builtins") match. Not stale as a fact about
that commit; will need a refresh after Inc 14.

## Findings

### L1 — Arena-overflow path for ARENA_PUSH_TRIPLE has no runtime test (LOW, conf 85)

- `helixc/tests/test_stage36_provenance.py:2263-2283`
- `test_stage36_inc14_register_derivation3_arena_overflow_returns_zero_handle`
  is honestly labeled "Structural test: we can't easily fill a 2M-slot
  arena, so this confirms the pure structural symmetry". The actual
  overflow code path in `x86_64.py` (the `mov eax, -1` + `jmp +24`
  sequence, lines 2761-2762) is NOT exercised by any test in this commit
  range. ARENA_PUSH_PAIR's equivalent overflow path has the same gap.
- Impact: a future codegen regression that mis-sized the `jmp +24`
  displacement or wrote to eax differently on the overflow side would
  pass the suite. Risk is low because the in-bounds tests do byte-exact
  exercise the rest of the layout (including all three writes + cursor
  bump), and the overflow path is 7 bytes of straight-line code.
- Suggested fix: a unit test that calls `register_derivation3` in a
  loop until cursor >= CAP-3 (likely needs an `__arena_set_cursor`
  helper or a Python-side test that drives the IR directly). Track as
  a follow-up; not blocking.

### L2 — register_derivation3 has no AD-erasability negative control (LOW, conf 82)

- `helixc/frontend/autodiff.py:75-96` + `helixc/tests/test_stage36_provenance.py`
- `register_derivation` was explicitly removed from `AD_KNOWN_PURE_CALLS`
  by Inc 11 so that `let _h = register_derivation(p, q);` inside a
  grad/grad_rev body fails closed with `NotImplementedError`. Inc 14
  added `register_derivation3` but did NOT add it to that set (correct
  by design), and there is no test pinning the analogous fail-closed
  behavior for the 3-arg variant.
- Impact: a future Inc could accidentally add `register_derivation3` to
  `AD_KNOWN_PURE_CALLS` (mirroring `parent_at` which IS in the set as
  a pure reader) and silently regress the same Inc 9 B2 / Inc 11 H1
  guarantee for the three-parent variant.
- Suggested fix: a 1-line test that calls `differentiate_reverse` on
  `let _h = register_derivation3(p, q, r); x` and asserts
  NotImplementedError with "cannot erase side-effecting" in the message
  — mirrors the existing `let_erasable_unused_and_logic_still_compiles`
  control (line 1770) but in the negative direction.

### L3 — has_evidence comment overstates the guarantee (LOW, conf 80)

- `helixc/stdlib/provenance.hx:22-34`
- The doc-comment claims "a handle that satisfies both predicates can
  be safely passed to evidence_left/evidence_right for use in
  downstream logic". This is true in the sense that the read returns
  a defined value (no UB). But the predicate cannot distinguish "the
  left slot was written by register_derivation(0, ...)" from "the left
  slot happens to hold a non-(-1) value from a different cause". With
  the current contract (left can legally be 0 or any positive i32),
  `has_evidence` is really an "is the slot in the recorded range"
  test, not an "is this a real handle" test.
- Impact: documentation-only. The function is sound; the bound is
  weaker than the comment implies. Misleading only if a downstream
  user reads the comment without reading the Phase-0 docs.
- Suggested fix: tighten the doc-comment to "returns 1 iff the
  handle's slot is within the recorded arena range; this is necessary
  but not sufficient for the handle to refer to a genuine
  register_derivation call (Phase-0 arena has no per-handle tag)".

---

**Verdict**: Ship as-is. The three Inc-12-catch-up/13/14 commits implement
their stated contracts correctly, the codegen byte-counts check out, the
atomicity invariant is preserved, and the test suite has real adversarial
content. The findings above are coverage-tightening for future-proofing,
not corrections to anything that's currently wrong.
