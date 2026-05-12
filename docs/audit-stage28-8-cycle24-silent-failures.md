# Stage 28.8 pre-29 audit gate — Cycle 24 (Audit A: silent failures)

**Date:** 2026-05-11
**HEAD:** `89d49e9` ("Audit 28.8 cycle 23 fix-sweep: close C22-1 + C22-3
(HIGH) + C22-2/4/5 (LOW)")
**Lens:** silent failures (Audit A)
**Streak counter at start:** 1/5 (cycle 23 was the first fully-clean
cycle on integrated state per the cycle-24 task brief).

> Note on HEAD mismatch: the task brief named `4bdc800`, but the local
> repo HEAD is `89d49e9`, which is exactly one commit forward and
> contains the cycle-23 fix-sweep (`effect_check.py` + tests). The
> audit was performed at `89d49e9`. The cycle-23 fix is therefore the
> sole production-code surface that has changed since the previous
> clean-cycle baseline, and it is folded into the rescan below as a
> first-class verification target.

---

## Scope

Strict-criterion read-only audit. Re-scan all surfaces cycle 23 cleared
plus the cycle-23 fix-sweep itself:

1. `helixc/frontend/ast_walker.py` and the four walker-refactor sites
   (`panic_pass`, `deprecated_pass`, `struct_mono.visit_expr`,
   `grad_pass._expr_has_grad`).
2. Stage 28.8.1 determinism (`x86_64._op_index` id(op) → fn/op index,
   `match_lower._FRESH_COUNTER` reset).
3. Isize/usize fix sites in `const_fold.py`, `x86_64.py`, `ptx.py`.
4. Cycle-23 fix-sweep delta: `helixc/ir/passes/effect_check.py`
   (FFI_CALL, ARENA_PUSH/SET, QUOTE, REFLECT_HASH, TILE_INDEX_STORE,
   TRACE_ENTRY/EXIT) and matching `tests/test_effect_check.py`
   regressions.
5. Wider scan of `helixc/{bootstrap,frontend,ir,backend,stdlib}/` for
   any new silent-failure surfaces.

No re-flagging of cycle 1-23 findings (per task brief).

---

## Priority 1 — Re-scan of cycle-23 fix-sweep (sole production diff)

### Diff inventory

`git diff 4bdc800..89d49e9 -- helixc/` returns exactly two files:

| File                              | Lines | Kind        |
|-----------------------------------|-------|-------------|
| `helixc/ir/passes/effect_check.py`| +34   | OP table + callees() |
| `helixc/tests/test_effect_check.py`| +69  | regression tests |

No other production code changed since cycle 23 HEAD.

### `OP_EFFECTS` extension

`effect_check.py:49-73` adds seven entries to the per-op effect table:

```
FFI_CALL       → {"ffi"}     # C22-1 (HIGH)
ARENA_PUSH     → {"arena"}   # C22-3 (HIGH)
ARENA_SET      → {"arena"}   # C22-3 (HIGH)
QUOTE          → {"reflect"} # C22-2 defense-in-depth (LOW)
REFLECT_HASH   → {"reflect"} # C22-4 defense-in-depth (LOW)
TILE_INDEX_STORE → {"tile_io"} # C22-5 defense-in-depth (LOW)
TRACE_ENTRY    → {"trace"}   # C22-4 defense-in-depth (LOW)
TRACE_EXIT     → {"trace"}   # C22-4 defense-in-depth (LOW)
```

All seven entries are pure additions to a `dict` — no existing entry was
modified, no entry was removed. The `own_op_effects(fn)` loop at
`effect_check.py:142-145` walks every op in every block and unions the
table lookup; the additions therefore propagate to the per-fn closure
without any other code change.

### `callees()` extension

`effect_check.py:158-166` adds a parallel branch for `FFI_CALL`,
matching the existing `CALL` and `MODIFY` branches. `target` is read
from `op.attrs`, isinstance-checked as `str`, and either added by name
or as the literal `"<indirect-ffi>"` sentinel. The sentinel never
appears in `module.functions`, so `compute_closure` (lines 192-200)
routes it through the same `closure[n].add("unknown")` path as the
existing `<indirect>` sentinel. No new silent path: an indirect FFI
call still surfaces as `"unknown"` effect, which any non-`unknown`-
declaring caller fails.

### Doctring versus table consistency

The module-level docstring (`effect_check.py:15-22`) enumerates four
effect labels (`io`, `modify_self`, `alloc`, `unknown`). The cycle-23
fix-sweep added five new label families (`ffi`, `arena`, `reflect`,
`tile_io`, `trace`) without updating the docstring.

This was considered as a potential finding and **rejected**: the
docstring lists are not authoritative — `OP_EFFECTS` is. There is no
runtime check that gates labels against the doctring enumeration. A
@pure violation reports labels straight out of `OP_EFFECTS` (line 229),
which the regression tests `test_c22_*` already pin. No silent failure
surface exists. Treated as a doc-only forward note (out of scope per
the read-only task brief).

### Effect-name collision check

The new labels (`ffi`, `arena`, `reflect`, `tile_io`, `trace`) do not
collide with existing effect labels (`io`, `modify_self`, `alloc`,
`unknown`, `network`, `rng`, `time`, `fs`), `META_ATTRS` keys
(`is_pub`, `is_pure`, `pure`, `kernel`, `is_extern`, `extern_abi`,
`device`, `partial`, `total`, `checkpoint`, `verifier`), or any reserved
attribute prefix. `declared_effects()` strips `effect:` prefix and emits
the bare label — a user-declared `@effect(ffi)` parses to attribute key
`effect:ffi`, declared-effect set `{"ffi"}`, which now correctly matches
the IR-derived closure when the body actually calls an extern fn.

### Test coverage

`test_effect_check.py` adds four targeted regressions:

- `test_c22_1_ffi_call_is_a_side_effect` — asserts FFI_CALL in
  `OP_EFFECTS` with the `"ffi"` label.
- `test_c22_3_arena_ops_are_side_effects` — asserts both ARENA_PUSH and
  ARENA_SET in `OP_EFFECTS` with `"arena"`.
- `test_c22_defense_in_depth_op_effects_complete` — asserts QUOTE,
  REFLECT_HASH, TILE_INDEX_STORE, TRACE_ENTRY, TRACE_EXIT all present.
- `test_c22_1_ffi_callee_appears_in_callees_set` — asserts the
  callees() FFI_CALL branch populates the target.

All 22 tests in `test_effect_check.py` pass at HEAD `89d49e9`. The
broader scan ran `test_effect_check.py`, `test_codegen_determinism.py`,
`test_struct_mono.py`, `test_const_fold.py`, and `test_ptx.py` (128
tests total) — all pass.

**Verdict for Priority 1: clean.** The fix-sweep adds defense-in-depth
without introducing new silent surfaces.

---

## Priority 2 — Cycle-23 surface re-scan at HEAD `89d49e9`

### Target 1 — `helixc/frontend/ast_walker.py`

File unchanged since cycle 22. `_TYPE_FIELD_NAMES` /
`_NON_NODE_FIELD_NAMES` skip-lists, `_is_ast_node` filter, and the
`visit()` / `generic_visit()` dispatch are byte-identical to cycle-23
state. `ASTVisitor.visit()` (line 180) still handles `None` explicitly
(`if node is None: return None`), so a missing optional child cannot
crash silently. Cycle-23 verdict carries: clean.

### Target 2 — `_op_suffix` and `id(op)` index table

`helixc/backend/x86_64.py:825-876` unchanged. The per-fn `_op_index`
dict is built at `FnCompiler.__init__` (lines 835-840) by linear walk;
`_op_suffix` (lines 854-876) looks up `id(op)` in the dict and returns
the deterministic `{fn_index}_{op_index}` form, with the
`{fn_index}_unk{id(op):x}` fallback only on miss. The fallback path is
intentionally loud (the embedded hex address surfaces non-determinism
in any byte-diff test) and is covered by
`test_codegen_determinism.test_codegen_determinism_byte_identical_*`.
Cycle-23 verdict carries: clean.

### Target 3 — `_FRESH_COUNTER` reset

`helixc/frontend/match_lower.py:52` unchanged. `_FRESH_COUNTER[0] = 0`
runs at every `lower_matches(prog)` entry. The two pinning regression
tests
(`test_match_lower_fresh_counter_resets_per_call`,
`test_match_lower_fresh_counter_state_visible`) pass. Cycle-23 verdict
carries: clean.

### Target 4 — Four walker refactors

`panic_pass.py`, `deprecated_pass.py`, `grad_pass.py`,
`struct_mono.py` all unchanged since cycle 23. The body-walk pipeline
`visit_expr` → `_BodyVisitor.visit` in `struct_mono.py` (lines 177,
199, 205) is intact; the `_BodyVisitor` (lines 133-173) with its five
overrides (`visit_Cast`, `visit_Name`, `visit_TileLit`, `visit_Let`,
`visit_ConstStmt`) is byte-identical to cycle-23 state. The dead
`visit_stmt` shim removed in cycle 22 is still gone (grep confirms
zero hits). Cycle-23 verdict carries: clean.

### Target 5 — isize/usize fixes (cycles 16-21)

All eight width-keyed sites enumerated in cycles 22-23 live in
`helixc/backend/x86_64.py` (`_is_i64_type` line 1005, `_is_u64_type`
line 1013, `wide_widths` line 1042), `helixc/ir/passes/const_fold.py`
(`_INT_BITS` line 43 with `isize: 64, usize: 64`), and
`helixc/backend/ptx.py` (`_ptx_type_str` line 157, `_DTYPE_SIZE` line
340, `_DTYPE_PTX_LOAD` line 343, `_ld_reg_prefix` line 355). None of
these files appears in the `4bdc800..89d49e9` diff. Cycle-23
enumeration is exact. Verdict carries: clean.

The pre-existing `.get(dtype, 4)` / `.get(dtype, "u32")` soft-fallback
pattern in `ptx.py:350-353` is the cycle-22 forward note (centralize
scalar-width predicate at Stage 29) — recorded, not flagged per the
no-re-flag rule.

### Target 6 — Deferred grad_pass rewriter + resolver

`helixc/frontend/grad_pass.py` unchanged. The cycle-22 status (v0.2
`ASTTransformer` work, not flaggable per task brief) carries unchanged.
Recorded, not flagged.

---

## Audit findings

**Cycle 24 silent-failures audit: CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

Key observations:

- The cycle-23 fix-sweep is the sole production-code delta since the
  cycle-23 baseline. The fix is pure additive: seven new entries in
  `OP_EFFECTS`, one new branch in `callees()`. No existing entry was
  modified or removed.
- The cycle-23 doc/table-consistency observation (new effect labels
  `ffi`/`arena`/`reflect`/`tile_io`/`trace` absent from the module
  docstring's enumerated list) was evaluated and rejected as a finding:
  the docstring is informational, `OP_EFFECTS` is authoritative, and
  the four new tests pin the table directly.
- All 128 audit-relevant tests (`test_effect_check.py`,
  `test_codegen_determinism.py`, `test_struct_mono.py`,
  `test_const_fold.py`, `test_ptx.py`) pass at HEAD `89d49e9`.
- No new silent-failure surfaces have been introduced since cycle 23.
- No previously-clean surface has regressed.

**Clean-cycle counter:** was 1/5 → **advances to 2/5.**

Three more consecutive clean cycles required to fire the Stage-29 gate.

---

## Out-of-scope per task instructions

- Effect-label docstring drift in `effect_check.py` (lines 15-22 list
  four labels, the table now carries nine families) is recorded as a
  forward note, not a finding. Read-only cycle.
- The Stage-29-class "centralize scalar-width predicate" refactor
  (forward note since cycle 17) — not a cycle-24 finding per the
  no-re-flag rule.
- The v0.2 `ASTTransformer` base class for grad_pass rewriter +
  resolver — known limitation per cycle 22, not a finding.
- Future-AST-author drift risk on `_TYPE_FIELD_NAMES` — forward note
  from cycle 22, not a finding.

---

## Files touched by this audit

None — read-only. Only this doc.

## Cross-reference

- Cycle 23 silent-failures (declared CLEAN, advanced 2/5 → 3/5):
  `docs/audit-stage28-8-cycle23-silent-failures.md`.
- Cycle 23 HEAD: `4bdc800`; cycle 24 HEAD: `89d49e9` (cycle-23
  fix-sweep commit).
- Production-code delta scope: `helixc/ir/passes/effect_check.py`
  (+34) and `helixc/tests/test_effect_check.py` (+69) only.
- Test suite verification: 128 audit-scope tests pass.
