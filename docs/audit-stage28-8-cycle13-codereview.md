# Stage 28.8 Pre-29 Audit Gate — Cycle 13, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: 98834de (read-only)
**Scope**: Audit C (general code-review) at commit 98834de.
HEAD is doc-only since c2e36d4: `git diff --stat c2e36d4..HEAD --
helixc/` returns empty (no production-code or test-code delta).
Per the cycle-13 prompt, this is the third consecutive
stability re-verification cycle on the same production HEAD,
and the rotation target is moved to a fresh module not covered
by cycle 11 (`check.py` / `parser.py` / `test_typecheck.py`)
or cycle 12 (`cse.py` / `const_fold.py`).

**Cycle-counter status going in**: 3/5 (cycles 10, 11, 12 all
CLEAN). This cycle, if clean across all three audits (A, B, C),
advances 3/5 -> 4/5.

**Rotation target chosen**: `helixc/ir/passes/dce.py` (130 LOC,
not previously deep-audited at Stage 28.8). Justification:

- Same pass family as the IR optimization passes named in the
  cycle-12 commit message as the intended rotation (CSE +
  const-fold). The cycle-12 codereview doc body in fact stayed
  on check.py / parser.py surface, so cse.py and const_fold.py
  are still un-rotated at audit-doc level. The cycle-13 walk-
  through performed an independent pass on cse.py +
  const_fold.py first (no findings at conf >= 80; documented
  internally below) before moving to dce.py as the next-fresh
  rotation target. Mental model of the IR + Value/Op datatypes
  carried cleanly across.
- `dce.py` carries a documented historical near-miss (the
  Stage 16.5 FFI_CALL audit CRITICAL-1, where DCE was silently
  dropping void-return extern calls because their results were
  never live). That comment at lines 60-64 was the explicit
  hint to look for analogous "kept-only-by-no-results" gaps in
  the SIDE_EFFECT_KINDS set.

**Method**:

(a) Read `docs/audit-stage28-8-cycle12-codereview.md` for
prior-cycle baseline + carryover concern list.

(b) Confirmed no production-code delta since c2e36d4 via
`git diff --stat c2e36d4..HEAD -- helixc/` (empty).

(c) Read `helixc/ir/passes/dce.py` end-to-end. Cross-referenced
SIDE_EFFECT_KINDS against the full `tir.OpKind` enum, computed
the complement, and for each op in the complement traced
whether the op (i) has results in practice and (ii) has side
effects beyond its operand graph.

(d) Probed the two no-result ops in the complement —
`TRACE_ENTRY` and `TRACE_EXIT` — by constructing a minimal
program that exercises them through the optimization pipeline,
then ran the result through `dce_module` + `compile_module_to_elf`
at `-O2` to observe end-to-end behavior.

(e) Re-confirmed cycles-6-through-12 below-threshold concerns
have not been promoted.

**Reporting threshold**: confidence >= 80 (strict criterion
per user directive 2026-05-10).

**Result**: **1 finding (1 HIGH at confidence 95) at or above
the confidence-80 reporting threshold.**

This breaks the cycle-10 / cycle-11 / cycle-12 clean streak.
**Cycle-counter does NOT advance.** Counter remains at 3/5;
under the strict-zero rule a non-clean cycle does not reset
the counter (cycles 10-12 are still recorded clean), but this
cycle does not credit toward the 5-clean-cycles gate, and a
fix-sweep + 5 fresh clean cycles are required to clear the
gate.

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| C13-1 | HIGH     | 95         | `helixc/ir/passes/dce.py:32-68`, `helixc/ir/lower_ast.py:567-583` | DCE drops the synthesized `const_int(0)` consumed by `TRACE_EXIT` in unit-returning `@trace` fns; backend crashes with `KeyError` at `_slot_of(op.operands[0])` (x86_64.py:2498) on `-O2`. |

---

## Finding C13-1 — DCE drops TRACE_EXIT's synthesized operand for unit-returning `@trace` fns

**Severity**: HIGH
**Confidence**: 95
**Component**: `helixc/ir/passes/dce.py` (the SIDE_EFFECT_KINDS
set, lines 32-68) + the interaction with the lowering pattern
at `helixc/ir/lower_ast.py:567-583` and the backend op-emitter
at `helixc/backend/x86_64.py:2489-2502`.

### What dce.py does

`dce_function` (dce.py:79-130) computes liveness in two
phases:

1. **Seed** (lines 87-99): for every op whose kind is in
   `SIDE_EFFECT_KINDS`, mark each of its operands' ids as live.
   Then mark all function params + block params live.

2. **Propagate** (lines 100-112): fixpoint — any op whose
   result is in `live` causes its operands to be added to
   `live`.

3. **Drop** (lines 114-129): for each op, keep it iff
   (a) its kind is in `SIDE_EFFECT_KINDS`, or
   (b) it has no results (`if not op.results`), or
   (c) at least one of its results is in `live`.
   Otherwise drop.

The third keep-arm (`not op.results`) is the load-bearing path
for `TRACE_EXIT` and `TRACE_ENTRY`: neither is in
`SIDE_EFFECT_KINDS` (verified by re-reading the set at
dce.py:32-68 and cross-checking against the
`tir.OpKind` enum), but both are emitted with no `result_ty=`
in `lower_ast.py` (lines 479, 575), so `op.results == []`,
and they survive the drop loop via clause (b).

The bug: **clause (b) keeps the op but does NOT seed its
operands as live.** The seed step (1) only walks
SIDE_EFFECT_KINDS' operands; the propagate step (2) only fires
when an op has a live result. An op kept solely by clause (b)
is in neither category — so any value that's consumed ONLY by
a clause-(b) op gets dropped, even though its sole consumer
survives. The result is a dangling operand reference: the
surviving op points at a `Value` whose producing op was just
deleted.

### How the producing case arises

`lower_ast.py:567-583`:

```python
# Audit 28.8 A7 — Stage 25 @trace epilogue. Emit TRACE_EXIT
# with the return value (so the runtime can record it) before
# the actual return instruction. If the fn returns Unit, pass
# a synthesized 0 sentinel.
if is_fn_traced:
    ret_operand = body_val
    if isinstance(ir_fn.return_ty, tir.TIRUnit) or ret_operand is None:
        ret_operand = self.builder.const_int(0)   # <- synthesized
    self.builder.emit(tir.OpKind.TRACE_EXIT, ret_operand,
                      attrs={"fn_name": fn.name})
# Emit return
if isinstance(ir_fn.return_ty, tir.TIRUnit):
    self.builder.ret(None)                        # <- RETURN with no operand
elif body_val is not None:
    self.builder.ret(body_val)
else:
    self.builder.ret(None)
```

For a unit-returning `@trace` fn (e.g. `@trace fn foo() {
let x: i32 = 5; }`), the lowerer:

1. Synthesizes a fresh `CONST_INT` op producing `Value v_k` (call it `v1`).
2. Emits `TRACE_EXIT(v1)` — no results.
3. Emits `RETURN()` — no operands.

The synthesized `v1` is consumed **only** by `TRACE_EXIT`. Since
`TRACE_EXIT` is not in SIDE_EFFECT_KINDS, `v1` is not in the
seed. Since `TRACE_EXIT` has no results, it never appears as a
"live-result" propagation source. `CONST_INT` is not in
SIDE_EFFECT_KINDS, has a result, and that result isn't live —
so DCE drops it. `TRACE_EXIT` survives but its `operands[0]`
now points at a `Value` whose producer is gone.

### Why the existing `@trace` tests miss this

`helixc/tests/test_trace.py` exercises four end-to-end trace
shapes (test_a7_trace_lowers_to_ir_entry_exit,
test_a7_backend_emits_trace_ops_as_stubs,
test_c2_2_early_return_emits_trace_exit,
test_c2_2_early_return_void). Every one of them declares the
`@trace`'d fn with an explicit i32 return type (`-> i32`), so
`is_fn_traced` is True but `isinstance(ir_fn.return_ty,
tir.TIRUnit)` is False and `body_val` is non-None. The
synthesized-const path at lower_ast.py:573-574 is never
taken. Additionally,
`test_a7_backend_emits_trace_ops_as_stubs` calls
`compile_module_to_elf` directly (which does NOT run
`dce_module` — only the `__main__` driver and `check.py`'s
`-O2` arm do, see x86_64.py:3097 and check.py:584), so even
if a unit-returning traced fn slipped past the lowerer, the
test's call site wouldn't exercise the DCE pass that drops
the const.

The strict-stdlib tests (test_strings_io, test_codegen) do
not declare any `@trace` fns at all.

`test_dce.py` covers DCE on plain arithmetic/CSE patterns but
does not construct any `@trace` fn. No test in the suite
covers the (trace-attr + TIRUnit-return + DCE-active) tuple.

### Reproducer

```python
from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck
from helixc.ir.lower_ast import lower
from helixc.ir.passes.dce import dce_module
from helixc.backend.x86_64 import compile_module_to_elf

src = '@trace fn foo() { let x: i32 = 5; }\nfn main() -> i32 { foo(); 0 }'
prog = parse(src)
typecheck(prog)
mod = lower(prog)
dce_module(mod)                  # <-- drops the CONST_INT producer
compile_module_to_elf(mod)       # <-- crashes here
```

Output (Python traceback):

```
File "helixc/backend/x86_64.py", line 2498, in _emit_op
    ret_slot = self._slot_of(op.operands[0])
File "helixc/backend/x86_64.py", line 864, in _slot_of
    return self.slots[v.id]
           ~~~~~~~~~~^^^^^^
KeyError: 1
```

End-to-end through the CLI driver:

```python
from helixc.check import main
# write src to a temp .hx file at <path>; output to <outpath>
rc = main([path, '-o', outpath, '-O2'])
```

Output:
```
helixc: internal error: KeyError: 1
helixc: this is a compiler bug ? please file an issue.
-- helixc-check: ...
   parse:    OK  (2 fns, 2 items)
   typecheck: OK
   totality:  OK
rc=1
```

At `-O1` (the default) the bug is latent: `check.py:579-588`
only runs `dce_module` when `a.opt_level >= 2`, so default-O1
compilation of the same program succeeds. The bug is therefore
contingent on `-O2` (or `-O3`, which is documented as identical
to `-O2`) + `@trace` + unit-return.

The lowered IR before DCE (probed via the same harness):

```
foo:
  trace.entry() {fn_name=foo}
  v0 = const.int() {value=5}    # the `let x = 5` literal
  v1 = const.int() {value=0}    # the synthesized TRACE_EXIT operand
  trace.exit(v1) {fn_name=foo}
  return()
```

After DCE:

```
foo:
  trace.entry() {fn_name=foo}
  trace.exit(v1) {fn_name=foo}  # v1 is dangling: no producing op left
  return()
```

The audit harness explicitly verified the dangling reference:
"DANGLING: trace.exit(v1) {fn_name=foo} references v1 which
has no producer".

### Static analysis: why the DCE-trace comment in x86_64.py
hints at this

The x86_64 backend at lines 2493-2494 says:

```python
# Phase-0: NOP stub (matching TRACE_ENTRY). The return
# value operand is still consumed so liveness analysis
# keeps the value alive past the trace call.
```

This comment encodes the author's assumption — that liveness
analysis WOULD keep the const alive past TRACE_EXIT. That
assumption is wrong: dce.py's liveness only walks operands of
ops in SIDE_EFFECT_KINDS, and TRACE_EXIT is absent from that
set. The backend's `mov_eax_mem_rbp(ret_slot)` at line 2500
exists to defeat register-allocator elision, but it can't
defeat upstream IR-level DCE if the producer is dropped
outright. The hazard is precisely the
"silently-dropping-void-return-op" pattern that the Stage 16.5
FFI_CALL audit (cited inline at dce.py:60-64) closed for
FFI_CALL — it just wasn't extended to TRACE_ENTRY / TRACE_EXIT
when those ops were added in cycle 2 / A7.

### Confidence: 95

Five independent pieces of evidence:

1. **Static**: cross-checked the full `tir.OpKind` enum
   against the SIDE_EFFECT_KINDS set; TRACE_ENTRY and
   TRACE_EXIT are the only no-result emit sites in
   `lower_ast.py` that aren't in the side-effect set.
2. **Dynamic IR**: constructed a minimal repro, ran
   `dce_module` on the lowered IR, observed `v1`'s producer
   dropped while TRACE_EXIT survived referencing `v1`. Output
   captured verbatim above.
3. **Dynamic backend**: ran `compile_module_to_elf` on the
   post-DCE IR, observed `KeyError: 1` at the exact
   `_slot_of(op.operands[0])` call site predicted by the
   static analysis (x86_64.py:2498).
4. **End-to-end CLI**: ran `helixc check ... -o ... -O2` on
   the source, observed `helixc: internal error: KeyError:
   1` + "compiler bug" message (rc=1).
5. **Cross-reference**: the comment at x86_64.py:2493-2494
   explicitly encodes the buggy assumption ("liveness analysis
   keeps the value alive past the trace call"), confirming
   this is an unintended hazard rather than deliberate
   contract.

5 points below 100 because: (a) bug is contingent on a
specific feature combo (`@trace` + unit-return + `-O2`) that
may not appear in real Phase-0 programs yet, so user-facing
impact is currently bounded; (b) the failure mode is a clean
crash with an attributable "compiler bug" message rather than
silent miscompilation, so production-data correctness is not
at stake.

### Suggested fix (NOT applied — audit is read-only)

Add `TRACE_ENTRY` and `TRACE_EXIT` to the SIDE_EFFECT_KINDS
set in `dce.py:32-68`. They have side effects by definition
(they communicate with the runtime trace buffer once the
runtime is linked) and their operands must remain live for
the backend's `_slot_of` lookup to succeed even when
Phase-0's no-op stub elides the actual call.

A regression test pinning the contract:

```python
def test_c13_1_dce_keeps_trace_exit_const_operand_for_unit_traced_fn():
    """C13-1: DCE must NOT drop the synthesized const that feeds
    TRACE_EXIT in a unit-returning @trace fn. Pre-fix, the
    dropped CONST_INT left TRACE_EXIT with a dangling operand
    that crashed the x86_64 backend at _slot_of."""
    from helixc.frontend.parser import parse
    from helixc.frontend.typecheck import typecheck
    from helixc.ir.lower_ast import lower
    from helixc.ir.passes.dce import dce_module
    from helixc.backend.x86_64 import compile_module_to_elf
    src = '@trace fn foo() { let x: i32 = 5; }\nfn main() -> i32 { foo(); 0 }'
    prog = parse(src)
    typecheck(prog)
    mod = lower(prog)
    dce_module(mod)
    elf = compile_module_to_elf(mod)   # MUST NOT raise
    assert isinstance(elf, (bytes, bytearray)) and len(elf) > 0
```

A second test covering the no-arg traced fn (`@trace fn bar() {}`)
should also pass post-fix; the lower_ast.py:572 path emits a
const_int(0) when `body_val is None`, same hazard shape.

---

## Below-threshold observations from this cycle's dce.py walk

The following items were re-rated by fresh eyes but remain
below the confidence-80 threshold and are NOT counted as
cycle findings. Surfaced here for cumulative carryover only.

### B13-1 — dce_function's outer `while changed` recomputes liveness from scratch each iteration (conf 55)

The outer fixpoint at dce.py:82-129 re-seeds + re-propagates
liveness on every iteration even though only deletions occur
between iterations. A standard implementation would worklist
the just-deleted ops' operands to see if any of their other
consumers are also dead — O(n) extra work amortized rather
than O(n*k) re-walks for k iterations. **Correctness is
unaffected.** A pathological deep dead-chain could be quadratic
in op count, but real programs have small chains. Performance
nit. Not promotable to >= 80 without a profiled regression on a
realistic IR. Conf 55.

### B13-2 — `TENSOR_STORE` declared in tir.OpKind but unreferenced anywhere (conf 35)

`tir.OpKind.TENSOR_STORE` exists at tir.py:138 but `grep` finds
zero emit sites or backend-handler arms across the entire
codebase. Not in SIDE_EFFECT_KINDS either. This is a
dead/aspirational enum entry, not a bug — the value-store ops
that actually fire are `STORE_VAR` and `STORE_ELEM`, both
properly in SIDE_EFFECT_KINDS. If a future feature wires
TENSOR_STORE without remembering to add it to
SIDE_EFFECT_KINDS, the same pattern as C13-1 could resurface
for tensor-store ops with no results. Conf 35 — speculative,
no current trigger.

### B13-3 — Lack of any DCE invariant test asserting "no op references a value whose producer was just dropped" (conf 50)

A defensive post-DCE pass that walks every op and asserts each
operand id is in `(producers + params + block_params)` would
have caught C13-1 immediately. This is a missing-fixture
observation, not a bug in dce.py itself. Conf 50 — a test
density concern. Promoting to a fix-sweep candidate would
be reasonable but not required by the audit threshold.

---

## Below-threshold re-evaluation (cycles 6-12 carryover)

Walked the cumulative below-threshold concern list once more.
Third- or later-pass for cycle-9 / cycle-10 surface area.

- **Cycle 6** (test additions for cycle-5 C5-1 TRAP constants;
  monomorphize_safe docstring drift): no change. Conf <= 40.
- **Cycle 7** (D-vs-Quote diagnostic text): no change. Conf 30.
- **Cycle 8** (C7-1 close test-coverage gap for `_compatible(
  TyMemTier, TyVar)` / `_compatible(TyMemTier, TySize)`):
  no change. Conf 55.
- **Cycle 9** (OSError edge cases not in the explicit set;
  pathological double-pre-prefix): no change. Conf 25-55.
- **Cycle 10 / 11 / 12** (pathological strip-shape coverage;
  end-to-end strict-stdlib coverage; UnicodeDecodeError
  single-prefix coverage; test ordering; signature; commit-
  message hygiene): no change. Conf 10-55.

No promotions to >= 80 from prior cycles. C13-1 is the only
new finding this cycle.

---

## Open prior findings (not addressed this cycle)

Per cumulative carryover from cycle 12:

- **audit-C4-1** (CRITICAL — D2 Call-RHS i32 SIGILL): still
  open. Out of scope for an audit-C stability cycle.
- **audit-C4-4** (HIGH — D9 paper-only): still open.
- **audit-C4-8 deferred** (LOW — check.py doesn't call
  fn-mono): still open.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open.
- **D-vs-Quote diagnostic text** (cycle-7 deferred): still open.
- **C7-1 close test-coverage gap** (cycle-8 housekeeping):
  still open.

C13-1 (above) is the new finding. Severity HIGH, confidence 95.

---

## Verdict

**Cycle 13 Audit C: NOT CLEAN — 1 finding (1 HIGH at
confidence 95) at or above the confidence-80 reporting
threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts
CLEAN only when zero findings of ANY severity at the audit
threshold. **This cycle does NOT qualify as clean.**

**Cycle-counter does NOT advance**: remains at 3/5. Cycles
10, 11, 12 stay credited; cycle 13 does not credit. To clear
the 5-clean-cycles gate, a fix-sweep closing C13-1 (with a
regression test, e.g. the one suggested above) must land,
followed by 5 consecutive clean cycles starting from the
fix-sweep HEAD.

The previously CLEAN production HEAD (c2e36d4 through 98834de,
unchanged at the source-code level) is now known to harbor an
`-O2`-conditional crash on a documented language feature
(`@trace` on unit-returning fns). Stage 29 (drop Python helixc)
must NOT proceed until C13-1 is closed: the bug is reachable
through a legal program, the crash surfaces as "compiler bug"
text rather than a clean diagnostic, and the fault is in the
optimization pipeline that Stage 29 inherits wholesale.

---

## Cycle 13 status

- Audit C result: **NOT CLEAN — 1 HIGH finding (C13-1) at
  confidence 95.**
- Counter going in: 3/5 (cycles 10, 11, 12 clean).
- Counter going out: **3/5 (no advance).**
- Production HEAD audited: 98834de (identical at source level
  to c2e36d4).
- Rotation target: `helixc/ir/passes/dce.py` (fresh — not
  covered by cycles 11 or 12).
- Stage 29 gating: **BLOCKED on C13-1 fix-sweep + 5 fresh
  clean cycles.**
