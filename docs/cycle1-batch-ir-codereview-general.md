# Cycle 1 Batch IR — General Code Review (Auditor 3)

Audit date: 2026-05-18
Auditor: pr-review-toolkit:code-reviewer

## Verdict

NOT_CLEAN — 2 HIGH + 9 MEDIUM/LOW

The IR subsystem is well-instrumented with audit-cycle commentary and
loud-fail discipline (raise vs. silent return). The two HIGH findings
below are correctness risks that survive the discipline at boundaries the
existing audit cycles did not cover. Most MEDIUM findings are sibling-pass
inconsistencies (e.g., one pass uses `raise RuntimeError`, the next still
uses `assert`) or dead helpers that audit docs flagged in prior cycles but
were never deleted.

---

## HIGH findings

### HIGH-1: `redundant_zero_coalesce` leaves dangling cross-block references

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\tile_opt.py`
**Lines**: 177-227 (loop body 196-226)

**Problem**: The pass collapses two TILE_ZEROS ops within a single block
by dropping the later op and rewriting *operand uses inside that same
block* (`remap.get(v.id, v)` at line 222). The docstring at lines 185-187
acknowledges only one cross-block constraint — that producers in different
blocks are never coalesced (per-block `seen` map). But it omits the dual
hazard: a TILE_ZEROS in block A whose result is also consumed by an op in
block B (a downstream block, or a backwards-flowing CFG arc). When block A
contains a redundant pair, the second TILE_ZEROS is dropped; consumers in
block B that referenced the dropped id now point at a value with no
producer.

In strict SSA the result id has exactly one definition. Removing it
without rewriting *every* consumer (in every block) is a use-without-def
in the lowered IR.

**Why it matters**: A TileFn that fans out a TILE_ZEROS to a downstream
block (e.g., a loop epilogue writing the accumulator back to HBM) will
miscompile silently: the backend either looks up a non-existent register
slot (KeyError) or, worse, reuses the slot for a different tile and emits
the wrong value. The Phase-0 status comment at lines 14-17 of the file
says "minimum-viable scope" and the regression may not yet manifest, but
the bug is real today — any subsequent test or kernel that uses TILE_ZEROS
twice in a block whose downstream block reads either one will trip it.

**Fix**: After per-block coalescing, run a second sweep over *all* blocks
in the function and apply the union of all per-block `remap` dicts to
their operand lists. Alternative: restrict coalescing to the within-block-
only consumers (skip when any other block references the result id) —
detectable by a pre-pass that builds the global use-set.

---

### HIGH-2: `_check_kind_coverage` exhaustiveness guard stripped under `python -O`

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\tile_opt.py`
**Lines**: 102-118

**Problem**: The module-load guard at lines 111-115 uses `assert` to
enforce that every `TileOpKind` is classified as either pure or
side-effecting. Under `python -O` / `PYTHONOPTIMIZE=1` / `.opt-1.pyc`
the assertion is stripped — a newly added TileOpKind would silently
default to "side-effect" (because `_SIDE_EFFECT_KINDS` is built as the
complement of `_PURE_TILE_KINDS` at line 97-99), which is conservative
for correctness but defeats the entire point of the guard ("Adding a
new TileOpKind without classifying it here is a build error", line
104-105).

The const_fold.py cycle 32 audit-R C31-3 fix (lines 64-78) and
effect_check.py cycle 32 audit-R C31-4 fix (lines 439-444) both
explicitly migrated their `assert` guards to `raise RuntimeError` /
`raise TypeError` for exactly this reason. The tile_opt.py guard is
the only one in the IR subsystem that wasn't migrated.

**Why it matters**: The discipline that protects sibling passes from
silent-failure regressions is inconsistent across the IR layer — a
contributor who reads const_fold or effect_check's hardened guards will
assume tile_opt has the same protection. Production releases built
with `-O` lose the guard entirely.

**Fix**: Replace lines 111-115 with an explicit `raise RuntimeError(...)`
loop, matching the const_fold and effect_check pattern:

```python
for k in ti.TileOpKind:
    if (k in _PURE_TILE_KINDS) == (k in _SIDE_EFFECT_KINDS):
        raise RuntimeError(
            f"TileOpKind {k.name} is missing or in both pure + side-"
            f"effect tables — classify it in tile_opt._PURE_TILE_KINDS"
        )
```

---

## MEDIUM findings

### MEDIUM-1: Dead helper `_find_value_by_id` in cse.py

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\cse.py`
**Lines**: 152-164

**Problem**: Zero callers in the codebase. Multiple audit-cycle docs
(`docs/audit-stage28-8-cycle16-silent-failures.md`,
`audit-stage28-8-cycle17-silent-failures.md`,
`audit-stage28-8-cycle18-silent-failures.md`) flag this as a code-hygiene
observation marked "clean / no action" — but it has now persisted across
three audit cycles. The helper's `return None` fallback is itself a
silent-failure shape; keeping it around invites a future caller to use
it and silently lose values.

**Fix**: Delete the function, or call it `_unused_find_value_by_id_dead_2026_05_18` and remove it next sweep.

### MEDIUM-2: Dead helpers `_is_int_const` and `_is_float_const` in const_fold.py

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\const_fold.py`
**Lines**: 247-252

**Problem**: Neither function is referenced anywhere in `helixc/`. Both
have parameters (`consts: dict`) that are never used inside the body —
they only check `op.kind`. They look like an abandoned earlier API where
`consts` was a separate constant-tracking dict.

**Fix**: Delete.

### MEDIUM-3: Dead helper `Lowerer._hash_ast` in lower_ast.py

**File**: `C:\Projects\Kovostov-Native\helixc\ir\lower_ast.py`
**Lines**: 4730-4733

**Problem**: Zero callers. The Quote arm at line 4655-4658 uses
`structural_hash` (with fallback to `_pretty`) directly; `_hash_ast` is
obsolete. The function's `abs(hash(_pretty(node))) & 0x7FFFFFFF` is a
weaker collision-prone hash than `structural_hash` — leaving it as
"dead but tempting" is a regression hazard.

**Fix**: Delete.

### MEDIUM-4: Unused import `struct` in const_fold.py

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\const_fold.py`
**Line**: 27

**Problem**: `import struct` is unused (only `struct.pack` appears in a
*comment* at line 34). Mild noise; would be flagged by any linter.

**Fix**: Delete the import.

### MEDIUM-5: `res_ty` dead local in const_fold._try_fold_op bitwise branch

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\const_fold.py`
**Line**: 566

**Problem**: `res_ty = op.results[0].ty if op.results else None` is
assigned at the top of the bitwise/shift try-block but never read. The
SHL and SHR branches each recompute the same value as `_res_ty` (lines
592, 606). Cosmetic but confusing — a maintainer who notices the
duplicate computation might "fix" it by reusing `res_ty`, which is
unguarded against `op.results == []` (line 566 returns None silently
where the SHL/SHR branches return None via the guard chain).

**Fix**: Delete line 566. The two `_res_ty` recomputes are intentional
guards for the empty-results case.

### MEDIUM-6: Layering violation — IR pass imports backend constant

**File**: `C:\Projects\Kovostov-Native\helixc\ir\lower_ast.py`
**Line**: 4640

**Problem**: `from ..backend.x86_64 import HELIX_NUM_CELLS` inside
`_lower_expr` introduces an `ir → backend` dependency that inverts the
standard compiler layering (`ir` should be backend-agnostic; `backend`
imports `ir`, not the reverse). Also, the import runs on *every* `quote`
lowering — small cost but unnecessary at hot-loop sites.

**Why it matters**: The QUOTE handle limit is a target-runtime
constraint, not an IR semantic. Phase-1 multi-backend work will need to
extract this constant somewhere `ir.passes` can reach without crossing
the layering line. Today's coupling makes that move harder.

**Fix**: Promote `HELIX_NUM_CELLS` to a shared location (e.g.,
`helixc/runtime/constants.py` or `helixc/ir/constants.py`) and import
from there in both backend and IR.

### MEDIUM-7: Silent miscompile — function-name reference lowers to const_int(0)

**File**: `C:\Projects\Kovostov-Native\helixc\ir\lower_ast.py`
**Lines**: 1730-1735

**Problem**: When `expr.name` resolves to a registered function (not a
local), the lowerer returns `const_int(0)` with the comment "v0.1: emit
a call-able marker". A user expression like `let f = my_fn; f(3)` would
compile with `f` bound to the integer 0, and any downstream code that
indirected through `f` would silently mis-execute. The `fn_alias` path
at lines 1733-1735 has the same shape.

**Why it matters**: This is one of the last remaining intentional
silent-miscompile points after Cycle 1 Batch IR's HIGH-1 fix (line
4723-4728) converted the catch-all to loud-fail. Function-as-value
is documented as v0.1-deferred but the placeholder of 0 is not
distinguishable at runtime from a legitimate i32 value of 0.

**Fix**: Either:
1. Raise `NotImplementedError` if `expr.name` is a function reference
   used in a value position (caller would need to pre-detect the
   indirect-call shape), or
2. Emit a `tir.OpKind.CONST_INT` with a sentinel attribute
   `{"is_fn_handle": True, "fn_name": expr.name}` so the backend can
   detect misuse and raise.

### MEDIUM-8: report_diagnostics severity label / prefix mismatch

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\effect_check.py`
**Lines**: 522-538

**Problem**: A `"hard"` severity prints with the literal `warning:`
prefix (line 525); the `"unknown"` branch (line 534-538) also uses
`warning:`. The cycle-32 C31-5 commentary explains this as
"grep-uniform with `warning:`" but the chosen vocabulary is confusing:
the *severity classification* is "hard" but the *user-visible label*
is "warning". A reader scanning compiler output for hard-fail
diagnostics would naturally search for `error:` first.

This matters because `report_diagnostics` returns `hard_count` and the
caller decides `--strict` abort semantics (line 489-490). The user
sees `warning:` lines that the compiler treats as errors when
`--strict` is on — surprising.

**Fix**: Change line 525 / 534 to `error: effect-check:` (or
`fatal: effect-check:`) so the textual label matches the abortable
severity tier. Keep the `warning:` shape only for `"info"` (which it
already overrides to `info:`).

### MEDIUM-9: `dce_function` recomputes liveness from scratch on every outer iteration

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\dce.py`
**Lines**: 106-153

**Problem**: The outer `while changed` loop (line 107) rebuilds `live`
from scratch each pass by re-scanning every block (lines 109-123) and
running an inner spread fixpoint (lines 125-136). The outer fixpoint
only re-runs when at least one op was dropped — so each outer iteration
is O(N) work for typically O(1) marginal change.

**Why it matters**: Compile-time concern only. On large functions this
is quadratic in the chain length of transitively-dead ops; for the
current Phase-0 IR sizes it's invisible, but the AGI-stack autodiff
pass produces deeply chained intermediates that may hit this.

**Fix**: Maintain `live` incrementally across outer iterations (only
re-spread when a previously-live op's operand becomes unreachable),
or do a single reverse-topological sweep computing transitive liveness
in one pass.

---

## LOW findings

### LOW-1: `import re` inside `live_function_names`

**File**: `C:\Projects\Kovostov-Native\helixc\ir\passes\fdce.py`
**Lines**: 41-42

**Problem**: `import re` and the `_ID_RE` compile happen inside
`live_function_names`, re-running on every fdce call. The pattern is
constant; should hoist to module scope.

**Fix**: Move both lines to module top.

### LOW-2: `FnIR.entry` and `TileFn.entry` raise opaque IndexError on empty `blocks`

**File**: `C:\Projects\Kovostov-Native\helixc\ir\tir.py` (line 423),
`C:\Projects\Kovostov-Native\helixc\ir\tile_ir.py` (line 140)

**Problem**: Both `@property entry` accessors return `self.blocks[0]`
with no guard. An empty `blocks` list raises a generic IndexError with
no context (which function? which module?). Phase-0 likely never hits
this, but the loud-fail discipline applied elsewhere in this audit
sweep suggests raising `RuntimeError` with the fn name on entry-block
absence is appropriate.

**Fix**:
```python
@property
def entry(self) -> Block:
    if not self.blocks:
        raise RuntimeError(f"FnIR {self.name!r} has no blocks; was begin_function() called?")
    return self.blocks[0]
```

---

## Notes

- `tir.py`, `effect_check.py`, and `dce.py` are well-commented with
  audit-cycle provenance (Stage NN cycle NN C-codes); the cross-references
  to docs/ are excellent and let an auditor trace decisions back to
  origin. This pattern should be the model for newer code.
- `const_fold.py`'s defense-in-depth exception narrowing (loud-fail
  re-raise of NotImplementedError / AssertionError / MemoryError before
  the catch-all `except Exception`) is exemplary and worth pulling into
  the IR-passes lint checklist.
- `cse.py` and `dce.py` are appropriately conservative; the pure-op set
  is small and explicit, the side-effect set is auditable.
- `effect_check.py` `META_ATTRS` and `META_ATTR_PREFIXES` membership is
  fragile — every new parser attribute requires a paired update here,
  with a trap 19002 false-positive as the failure mode. Worth a future
  refactor where the parser tags attributes as `effect` vs `meta`
  directly rather than relying on this allowlist.
- The cross-pass inconsistency in `assert` → `raise` migration (HIGH-2)
  is the most impactful pattern fix: every IR pass should adopt the
  cycle 32 C31-3/C31-4 hardening uniformly.
