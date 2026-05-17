# Stage 36 closure gate-3 silent-failure audit

**HEAD**: 2a6aedd (Stage 36 CLOSED 2026-05-16 at Inc 16) — helixc/ tree
identical to 97dbfbc per `git diff --stat 97dbfbc..HEAD -- helixc/`
(zero changes). Brief specified 97dbfbc; both audit the same tree.
**Scope**: `git diff e7c3552..HEAD -- helixc/` (Inc 15 + gate-1 fix
sweep + gate-2 test additions).
**Date**: 2026-05-16
**Auditor**: silent-failure lane

## Findings

### H1: `parent_right_at(0)` silently leaks `arena[0]` instead of returning -1 sentinel (HIGH, confidence 95/100)

**Location**: `helixc/ir/lower_ast.py:2145-2155` (the `parent_right_at`
lowering arm, unchanged by Inc 15 but adjacent to and explicitly
referenced by the Inc 15 changes).

**Pattern**: The Inc 15 sweep added an explicit `handle <= 0` runtime
guard to `parent_at` (lines 2192-2195) precisely to defeat the audit's
hidden-error #3 ("`parent_at(0, 1)` silently reading `arena[0]`"). The
symmetric hazard exists in the sibling `parent_right_at` and was NOT
patched. Because `parent_right_at` lowers to `_safe_arena_get(handle-1, +1)`
and `_safe_arena_get` only bounds-checks the FINAL effective index
(`eff_idx = (handle-1) + 1`), a null handle 0 produces `eff_idx = 0`,
which is in-bounds whenever any derivation has been registered, and the
function returns `arena[0]` — the first registered derivation's LEFT
value.

**Evidence** (reproducer run against HEAD = 2a6aedd):
```
fn main() -> i32 {
    let _h = register_derivation(11, 22);
    parent_right_at(0)        // expected -1 (255 as u8 exit), got 11
}
```
Exit code observed: **11** (would be 255 if the sentinel guard fired).
Same bug propagates through the stdlib readability alias
`evidence_right(0)` (also returns 11), and through the diagnostic helper
`trace_evidence(0)` whose stdout becomes `h=0 slot0=-1 slot1=11\n`
instead of the asymmetric `h=0 slot0=-1 slot1=-1\n` printed when the
arena is empty.

Critically, the existing canary `test_stage36_inc13_trace_evidence_returns_zero_for_null_handle`
(test file line 1942) has a docstring asserting `trace_evidence(0)`
prints `"h=0 slot0=-1 slot1=-1\n"`, but the test body discards stdout
(`rc, _stdout = _run_elf_capture(elf)`) and only checks the return
value. The aspirational docstring is not enforced; the bug ships
hidden behind a passing test. The Inc 15 stdout-format canary
`test_stage36_inc13_trace_evidence_stdout_format` (line 1957) only
covers the non-null `register_derivation(11, 22)` case, which happens
to work because the handle is valid (h=1 → slot1 read is arena[1]=22).

**Impact**: Users of the post-Inc-15 provenance helpers will reasonably
expect that `parent_right_at(null_handle)` returns -1 — every other
member of the family does (`parent_left_at(0)` correctly returns -1
because `-1+0=-1` is OOB; `parent_at(0, slot)` correctly returns -1
post-Inc-15 because of the new explicit guard). The asymmetry means
audit/debug code branching on "is the right-parent present?" using
`evidence_right(h) == -1` will incorrectly conclude there IS a
right-parent (with value 11, or whatever happens to live at arena[0])
for a null handle whenever the arena is non-empty. This is the exact
silent-failure pattern that justified the H1 closure work on
`parent_at`, applied incompletely.

**Fix**: Mirror the Inc 15 `parent_at` `handle <= 0` SELECT guard on
the `parent_left_at` and `parent_right_at` lowering arms. Concretely
in `lower_ast.py` after line 2150 (and similarly after 2143 for
`parent_left_at`, even though that one happens to work by accident —
defence in depth + symmetry with the typecheck strict-i32 family
tightening):

```python
idx = self._lower_expr(expr.args[0])
if idx is None:
    return None
zero = self.builder.const_int(0)
one = self.builder.const_int(1)
neg_one = self.builder.const_int(-1)
handle_valid = self.builder.emit(
    tir.OpKind.CMP_GT, idx, zero,
    result_ty=tir.TIRScalar("i32"))
base_idx = self.builder.emit(
    tir.OpKind.SUB, idx, one,
    result_ty=tir.TIRScalar("i32"))
raw = _safe_arena_get(base_idx, 1)   # 0 for parent_left_at
return self.builder.emit(
    tir.OpKind.SELECT, handle_valid, raw, neg_one,
    result_ty=tir.TIRScalar("i32"))
```

Add two test canaries to `test_stage36_provenance.py`:

1. `test_stage36_inc16_parent_right_at_null_handle_returns_neg_one` —
   register one derivation, call `parent_right_at(0)`, assert exit 255
   (or equivalent sentinel check).
2. Strengthen the existing `test_stage36_inc13_trace_evidence_returns_zero_for_null_handle`
   to actually assert on `stdout` AFTER a prior `register_derivation`,
   so the docstring's claim is enforced.

This is a HIGH because (a) Inc 15 explicitly framed the H1 work as
"defeating hidden-error #3 — null-handle silent leak", (b) the same
class of bug remains live in the sibling primitive that the user-facing
stdlib helper `evidence_right` delegates to, (c) there is no
typecheck-level catch (typecheck only constrains the type, not the
value), and (d) the existing test docstring asserts the correct
behaviour without enforcing it — a maintenance trap.

---

### M1: Strict-i32 family tightening rejects refinement-typed handles that pre-Inc-15 accepted (MEDIUM, confidence 80/100)

**Location**: `helixc/frontend/typecheck.py:2950-2967` (parent_left_at /
parent_right_at strict check) and:2989-3012 (parent_at strict check —
pre-existing in Inc 14, but the family is now uniformly affected).

**Pattern**: The new strict check is
`isinstance(arg_tys[0], TyPrim) and arg_tys[0].name == "i32"`. It does
NOT erase the Stage 31 `TyRefined` wrapper. Pre-Inc-15 the call went
through `_is_int_scalar` which calls `_erase_refinement(t)` first
(typecheck.py:6071). The semantic intent of refinement aliases (e.g.
`type PosHandle = i32 where self > 0`) is "the same i32, with extra
proof obligations" — exactly the right type for a derivation handle.
The post-Inc-15 strict check unintentionally rejects these.

**Evidence**:
```
type PosHandle = i32 where self > 0;
fn read_left(h: PosHandle) -> i32 { parent_left_at(h) }
fn main() -> i32 { read_left(1) }
```
Typecheck error: `parent_left_at(idx): arg must be exactly i32
derivation handle, got PosHandle (pre-Inc-15 also accepted i64/u32/u64
but those silently truncated in downstream arena read)`.

**Impact**: Loud, not silent — it's a hard typecheck reject, the
opposite of a silent failure per the brief's definition. Not flagged
as a silent-failure finding but documented here because the diff
introduces the regression. A future user who refines their handles
for additional safety (the intended Stage 31 idiom) will hit a wall.

**Fix**: Replace the `isinstance(arg_tys[0], TyPrim)` checks with
calls that first erase the refinement:
```python
erased = self._erase_refinement(arg_tys[0])
if not (isinstance(erased, TyPrim) and erased.name == "i32"):
    self.errors.append(...)
```
Apply uniformly to `parent_left_at`, `parent_right_at`, `parent_at`
(both arg positions) for family consistency. The error-message
prose ("pre-Inc-15 also accepted i64/u32/u64") would also
no longer be misleading — it's currently technically true but
masks the fact that refinement-i32 was *also* accepted pre-Inc-15
and is no longer.

This is MEDIUM not HIGH because (a) refinement use on parent helpers
is presumably rare today and (b) the failure mode is loud. But for
the closure gate it's a regression introduced by Inc 15 that the
brief's "type-design M1" framing did not anticipate.

---

## Considered but not flagged

### `evidence_third` / `evidence_middle` on a 2-parent handle

The provenance.hx doc-comments are explicit that `evidence_third(h)`
on a 2-parent handle "reads into whatever happens to live at the slot
after the right value (typically the next derivation's slot[0] or the
OOB sentinel)". This is the cross-record hazard, explicitly tracked
as the deferred `stage36-inc16-arity-in-handle` TODO. The Inc 15 ship
notes (in test docstrings and lower_ast comments) acknowledge it as
out of scope; the audit brief also says "this audit-fix increment"
cannot close cross-record. Not flagged — the deferral is documented
and called out at every code site.

### `trace_evidence` label change "L=/R=" → "slot0=/slot1=" — caller breakage

I grep-searched all .hx files and all .py files for the old `L=` / `R=`
literal format. No production callsite or non-Inc-13 test asserts on
the old format. The Inc 15 diff updates the two affected tests
(lines 1974, 1998 of test_stage36_provenance.py) atomically. Not flagged.

### `parent_at` SELECT chain evaluates `raw_read` unconditionally

The lowered IR computes `raw_read = _safe_arena_get(eff_idx, 0)` even
when guards fail. `_safe_arena_get` has its own internal clamp
(safe_idx = clamp(eff_idx, 0, arena_len-1)) so the speculative
ARENA_GET cannot fault. The outer SELECT then suppresses the value.
No correctness issue. Not flagged.

### BIT_AND on CMP_* results being non-0/1

x86_64.py:2168-2197 lowers CMP_* via `setl_al`/`setg_al`/etc., which
zero-extend a 0/1 byte into the result. BIT_AND on clean 0/1 produces
clean 0/1. SELECT branches correctly. Not flagged.

### parent_left_at(0) returning -1 by accident

This works only because offset=0 lets `eff_idx = -1` fall OOB. The
analysis above flags it for defence-in-depth in the H1 fix, but
left alone it is not currently a silent failure — the sentinel emerges
from the accidental composition with `_safe_arena_get`. Strictly
speaking, the test suite's existing
`test_stage36_inc14_parent_at_null_handle_returns_neg_one_sentinel`
also pins this for parent_at, but no equivalent test pins
`parent_left_at(0)` directly. Worth a defensive canary alongside
the H1 fix, but the existing behaviour is correct.

---

## Summary

**Total findings**: 1H, 1M, 0L.

The H1 finding is a concrete, reproducible silent failure that
contradicts the framing of the Inc 15 H1-closure work itself: Inc 15
added a `handle <= 0` runtime guard to `parent_at` to "defeat the
audit's hidden-error #3 (parent_at(0, slot) silently reading arena[0])",
but the structurally identical bug in the sibling `parent_right_at`
primitive — which is reached by the user-facing stdlib alias
`evidence_right` and by the diagnostic helper `trace_evidence` — was
not patched. An existing test docstring claims the correct behaviour
without asserting on the relevant output, so the bug ships behind a
green test. The fix is a 10-line mirror of the existing parent_at
guard plus two test canaries; the cumulative diff is otherwise solid
(strict-i32 family tightening is correctly motivated by the silent-
truncation hazard for i64 handles; the static + runtime slot bounds
on parent_at are well-decomposed between typecheck and lowering; the
provenance.hx doc rewrites honestly call out the slot-positional
sharp edges).

**Recommended action**: Block the 3/3 clean gate on H1. Land the
parent_left_at/parent_right_at null-handle guards + two canary tests,
then re-run gate-3.
