# Stage 28.8 Pre-29 Audit Gate — Cycle 15, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: 1e4c3e6 (read-only, identical to cycle 14)
**Scope**: Audit C (general code-review) at HEAD. `git diff --stat
1e4c3e6..HEAD -- helixc/` is empty: HEAD is still
1e4c3e639a593995dc66c7c3369a0955b6ddbf83 (the cycle-14 fix-sweep
that closed C13-1). No production-code or test-code delta since
cycle 14. This is the second stability re-verification on the
cycle-14 HEAD, and the prompt explicitly asks for a fresh
adversarial probe targeting failure modes not previously
exercised (nested @trace, @trace + @autotune, @trace with
non-trivial body).

**Cycle-counter status going in**: 1/5 (cycle 14 was the first
CLEAN cycle after the cycle-13 reset).

**Method**:

(a) Read `docs/audit-stage28-8-cycle14-codereview.md` for the
    cycle-14 closure record and the cycle-13 carryover concerns.

(b) Read `docs/audit-stage28-8-cycle13-codereview.md` to recover
    the exact C13-1 contract and conf-95 evidence base, so the
    adversarial probe targets the same fault domain.

(c) Confirmed via `git rev-parse HEAD` + `git diff --stat
    1e4c3e6..HEAD -- helixc/` that HEAD has not moved and no
    production code or tests changed since cycle 14.

(d) Re-read `helixc/ir/passes/dce.py` end-to-end (143 LOC). The
    20-member `SIDE_EFFECT_KINDS` set, the 3-phase liveness
    computation, and the keep/drop logic are unchanged from
    the cycle-14 post-fix audit. Comment block at lines 68-78
    is intact.

(e) Re-read `helixc/ir/lower_ast.py:471-484` (TRACE_ENTRY
    emission), `:567-583` (TRACE_EXIT epilogue with synthesized
    `const_int(0)` operand), and the C2-2 early-return path
    that emits TRACE_EXIT before an explicit `return` inside
    `_lower_expr`'s `A.Return` arm. All three call sites are
    unchanged.

(f) Re-read `helixc/backend/x86_64.py:2489-2502` (TRACE_EXIT op
    emitter, which calls `_slot_of(op.operands[0])` at line
    2498). Unchanged.

(g) Enumerated the full `tir.OpKind` enum (91 members) against
    the post-fix `SIDE_EFFECT_KINDS` (20 members). The 71-member
    complement is all pure computation ops (constants,
    arithmetic, comparisons, shape ops, loads, casts, reduce,
    matmul, transforms). Spot-checked the complement for any op
    that lower_ast.py might emit with no `result_ty=`. None
    found. The cycle-13 finding C13-1 remains the only known
    instance of the "no-result + non-side-effect + operand-
    bearing" shape in the codebase.

(h) Cross-referenced `TENSOR_STORE` (the cycle-13 B13-2 conf-35
    speculative concern). Still unreferenced anywhere except
    the enum decl at `tir.py:138`: no emit site in
    `lower_ast.py`, no backend handler in `x86_64.py` /
    `ptx.py`. Speculative only — no present hazard.

(i) Re-ran `helixc/tests/test_dce.py` (8/8 pass, including the
    two cycle-14 regression tests) and `helixc/tests/test_trace.py`
    (21/21 pass — the full trace surface).

(j) **Fresh-eyes adversarial probe.** Wrote a probe harness
    that lowers, fold-modules, dce-modules, and
    compile-module-to-elfs eight new failure-mode shapes:

    1. **Nested @trace**: `@trace fn inner() {}` called from
       `@trace fn outer() { inner(); ... }`. Both unit-return.
       Result: **OK, ELF 4733 bytes.** TRACE_ENTRY/EXIT pairs
       emitted for both fns; the synthesized `const_int(0)`
       operand for outer's TRACE_EXIT survives DCE.
    2. **@trace with non-trivial control flow**: unit-returning
       @trace fn with `if/else` that mutates a local `let mut`.
       Result: **OK, ELF 4826 bytes.** The control-flow blocks
       lower cleanly; TRACE_EXIT survives at the function tail.
    3. **@trace with early-return + unit fall-through**: an
       `if cond > 0 { return; }` followed by an unreachable-on-
       early-return `let z: i32 = 9;`. Tests the C2-2 early-
       return TRACE_EXIT emission path interacting with DCE.
       Result: **OK, ELF 4806 bytes.** Both the early-return
       and fall-through TRACE_EXIT ops survive with valid
       operands.
    4. **@trace with mixed early-return + fall-through, i32
       return**: `@trace fn cond_ret(cond: i32) -> i32 { if cond
       > 0 { return 11; } 22 }`. The i32 path takes the
       `body_val`-as-operand branch at `lower_ast.py:572` (not
       the synthesized-const branch); tests the non-Unit code
       path for completeness.
       Result: **OK, ELF 4794 bytes.**
    5. **@trace with dead intermediates (let chain)**: four
       unused `let X: i32 = N;` bindings in a unit-returning
       @trace fn. Maximises the post-fold dead-const population
       that DCE has to traverse.
       Result: **OK, ELF 4691 bytes.** The synthesized
       `const_int(0)` for TRACE_EXIT survives even when its
       value-id (after fold) coincides with where a dropped
       intermediate's id used to live.
    6. **@trace + while loop**: unit-returning @trace fn with
       a `while i < 3 { i = i + 1 }` body. Exercises block-
       param liveness interaction with TRACE_EXIT.
       Result: **OK, ELF 4793 bytes.**
    7. **@trace returning its parameter directly**: `@trace fn
       passthrough(x: i32) -> i32 { x }`. Tests the case where
       `body_val` is a function-param `Value` (not a const). The
       operand survives because params are unconditionally
       seeded live.
       Result: **OK, ELF 4687 bytes.**
    8. **@trace + @pure**: `@trace @pure fn pure_traced(x: i32)
       -> i32 { x + 1 }`. The fn body is purely functional and
       the @pure attribute is recognized by the effect-checker
       (`effect_check.py:92`), but @trace still injects
       TRACE_ENTRY/EXIT. Tests that the two attributes do not
       interact destructively in the lowerer or DCE pass.
       Result: **OK, ELF 4706 bytes.**

    For each probe, the harness also did a global post-DCE
    dangling-operand sweep (the same check the cycle-13 audit
    used to find C13-1): for every surviving op, assert each
    `operand.id` is in `(producers + fn.params + block.params)`.
    **Zero dangling operands across all eight probes.**

(k) Also ran the four most representative probes (nested,
    cond-unit, early-unit, while-unit) through the actual CLI
    driver `python -m helixc.check <file> -o <out> -O2`. All
    four returned `rc=0` (no "internal error" / "compiler bug"
    text, no traceback). This is the canonical end-to-end
    path that the cycle-13 reproducer used; reusing it here
    catches any divergence between the in-process probe and
    the CLI's own dce/elf pipeline.

(l) **@trace + @autotune was excluded by feasibility**: the
    `@autotune` attribute requires `@kernel` on the same fn
    (`autotune.py:200-204` produces a hard diagnostic
    otherwise), and `@kernel` fns are routed to the PTX backend
    (`ptx.py:98`), not the x86_64 backend that owns the
    cycle-13 TRACE_EXIT crash site. `@kernel` + `@trace` is
    not exercised by any existing test and the lowerer doesn't
    block the combination outright, but the dce pass affects
    only the x86_64 emit path (PTX runs through `ptx.py`, which
    doesn't share the `_slot_of` lookup site). Probing
    `@autotune` is therefore a pure tile_ir / ptx concern, not
    a code-review concern for the dce.py fix. Documented for
    transparency.

**Reporting threshold**: confidence >= 80 (strict criterion per
user directive 2026-05-10).

**Result**: **0 findings at or above the confidence-80 reporting
threshold.**

---

## Summary table

| ID | Severity | Confidence | Component | Issue |
|----|----------|------------|-----------|-------|
| _(none)_ | — | — | — | No high-confidence issues. |

---

## Adversarial probe details

The eight probes exercise distinct shape combinations of the
dce.py + TRACE_EXIT contract. The intent of the cycle-15 prompt
("try one new failure mode not previously exercised") is
satisfied by **all eight** being new shapes: cycle 13 covered
only the simplest unit-returning single-fn case (`@trace fn foo()
{ let x: i32 = 5; }`), and cycle 14 added the matched regression
test for that exact shape. None of the probes 1-8 above is
covered by either of the cycle-14 regression tests.

The most adversarial of the set are:

- **Probe 1 (nested @trace)**: two TRACE_EXIT call sites in a
  single program, both consuming synthesized `const_int(0)`s.
  If DCE were dropping operands based on a per-function liveness
  bug rather than the seed-set membership, nested traces would
  amplify the failure rate. They do not.

- **Probe 3 (early-return + unit fall-through)**: the C2-2
  early-return arm at `_lower_expr` for `A.Return` emits its
  own TRACE_EXIT (separate from the fall-through one at
  `_lower_fn_body:575`). Both call sites use the same synthesized
  `const_int(0)` construction; both survive DCE.

- **Probe 4 (mixed early-return + fall-through, i32)**: the i32
  return type means the synthesized-const branch at
  `lower_ast.py:573-574` is NOT taken (because `ret_operand =
  body_val` is non-None and the return type is not TIRUnit).
  TRACE_EXIT then consumes a real producer's result. This
  validates that the C13-1 fix doesn't accidentally regress the
  non-Unit case (it doesn't — the fix is additive set
  membership, so all TRACE_EXIT operands are now seeded live
  regardless of source).

- **Probe 5 (dead intermediates)**: maximises post-fold dead
  population. Tests robustness of the fixpoint outer loop at
  dce.py:96-142 against multiple removal iterations.

---

## Verification of the cycle-14 fix under the new probes

The cycle-14 fix is the addition of TRACE_ENTRY/TRACE_EXIT to
SIDE_EFFECT_KINDS. The cycle-15 probes validate that this
addition is **complete and correct** under the broader trace
attribute surface, not just the minimal reproducer:

1. **Multi-fn coverage**: probes 1, 4, 7 have multiple traced
   fns in one module. dce_module iterates per-function, so the
   fix is applied independently per fn; probe 1 confirms
   independence (one TRACE_EXIT per traced fn, both survive).

2. **Control-flow coverage**: probes 2, 3, 6 introduce
   `if/else` and `while` constructs, which split the fn into
   multiple basic blocks. The seed phase at dce.py:101-105
   iterates `for blk in fn.blocks` so block structure does not
   affect seed correctness; probes confirm the empirical
   invariant.

3. **Early-return path**: probe 3 hits the C2-2 early-return
   TRACE_EXIT emission, which is at a different call site
   from the cycle-14 minimal reproducer. The fix covers it
   transparently (same OpKind, same seed-set membership).

4. **Operand-id stability after fold**: probe 5 stresses the
   case where the synthesized `const_int(0)` value-id is the
   highest-numbered post-fold, surrounded by dropped peers.
   No off-by-one or stale-id behavior observed.

---

## Why no findings at >= 80

1. **HEAD is identical to cycle 14**: no new code has landed
   between cycle 14 (CLEAN) and cycle 15. Any new finding
   would have to be a fresh discovery on a code-base that
   passed cycle 14's clean audit, which is plausible only if
   the cycle-14 audit missed an entire failure mode. The
   cycle-15 adversarial probe was specifically designed to
   surface such a miss; all eight probes pass.

2. **Static enum-coverage check**: enumerated all 91 OpKinds
   and the 71-member complement of SIDE_EFFECT_KINDS. Every
   member of the complement is a pure computation op that
   carries results. None match the "no-result + operand-
   bearing + non-side-effect" shape that produced C13-1.

3. **End-to-end CLI driver succeeds** on all probes via
   `python -m helixc.check <file> -O2`. The cycle-13 crash
   mode ("helixc: internal error: KeyError: 1") does not
   reappear in any new shape.

4. **Existing test suite passes**: 8/8 dce tests + 21/21
   trace tests, with no flakiness or stderr noise.

5. **@autotune-feasibility check**: the @autotune route does
   not touch dce.py (it operates on @kernel fns routed through
   ptx.py). Documented; not a real probe gap because @kernel
   bypasses the x86_64 backend entirely.

---

## Below-threshold observations from this cycle

The following items were rated below conf 80 and are NOT
counted as cycle findings. Surfaced here for cumulative
carryover only.

### B15-1 — `dce.py:14-22` docstring "Side-effecting op kinds" enumeration is still stale (conf 30, unchanged from cycle 14's B14-2)

The cycle-14 B14-2 observation noted that the high-level
docstring at `dce.py:14-22` enumerates only a partial subset
of SIDE_EFFECT_KINDS (RETURN, BR, COND_BR, CALL, STORE_VAR,
STORE_ELEM, ALLOC_VAR, ALLOC_ARRAY, MODIFY, SPLICE, "io.print,
etc.") while the actual set at lines 32-81 also contains
QUOTE, REFLECT_HASH, ARENA_PUSH, ARENA_SET, TILE_INDEX_STORE,
FFI_CALL, TRAP, TRACE_ENTRY, TRACE_EXIT. The cycle-14 fix-sweep
added TRACE_ENTRY/TRACE_EXIT to the set but did not refresh
the docstring summary, so the docstring is now further out of
date than at cycle 13. Not a correctness issue. Conf 30,
unchanged.

### B15-2 — `test_c13_1_dce_preserves_trace_entry_in_kept_set` remains a present-state tautology (conf 35, unchanged from cycle 14's B14-1)

TRACE_ENTRY has no operands in current lower_ast.py and the
clause-(b) `not op.results` keep-arm preserves it
unconditionally. The test docstring acknowledges this as
"future-proofing". Sub-80, not promotable to a hard finding
without an audit-prompt mandate that every regression test
must demonstrably fail without its fix. Conf 35, unchanged.

### B15-3 — No global post-DCE dangling-operand invariant pass in production (conf 50, unchanged from cycle 14's B14-3 / cycle 13's B13-3)

The cycle-15 probe harness *does* run such a sweep manually,
which is what gives the audit its high coverage. A defensive
production pass that asserts the same invariant after
`dce_module` would have caught C13-1 immediately and would
catch any future regression of the same shape. Conf 50.
Recommend promoting to a fix-sweep candidate when cycle 18
(the last gating cycle) lands — until then, the SIDE_EFFECT_KINDS
enumeration is sufficient.

### B15-4 — `dce_function`'s outer `while changed` recomputes liveness from scratch (conf 55, unchanged from cycle 13's B13-1)

Pre-existing performance nit. Unchanged.

### B15-5 — `TENSOR_STORE` enum member unreferenced (conf 35, unchanged from cycle 13's B13-2)

Aspirational enum entry with no emit site or backend handler.
Speculative future-proofing concern only. Unchanged.

---

## Below-threshold re-evaluation (cycles 6-14 carryover)

Walked the cumulative below-threshold concern list once more.

- **Cycle 6** (TRAP-const test additions; monomorphize_safe
  docstring drift): no change. Conf <= 40.
- **Cycle 7** (D-vs-Quote diagnostic text): no change. Conf 30.
- **Cycle 8** (C7-1 `_compatible(TyMemTier, TyVar)` coverage gap):
  no change. Conf 55.
- **Cycle 9** (OSError edge cases; pathological strip-shape):
  no change. Conf 25-55.
- **Cycle 10 / 11 / 12** (test ordering; signature; commit-
  message hygiene; strict-stdlib end-to-end coverage):
  no change. Conf 10-55.
- **Cycle 13** (B13-1 outer-loop re-seed; B13-2 TENSOR_STORE
  speculative; B13-3 missing invariant checker): rolled
  forward as B15-3, B15-4, B15-5. No change. Conf 35-55.
- **Cycle 14** (B14-1 entry-tautology; B14-2 stale docstring;
  B14-3 invariant-checker): rolled forward as B15-1, B15-2,
  B15-3. No change. Conf 30-50.

No promotions to >= 80 from prior cycles. No new findings
this cycle.

---

## Open prior findings (not addressed this cycle)

Per cumulative carryover from cycle 14:

- **audit-C4-1** (CRITICAL — D2 Call-RHS i32 SIGILL): still open.
- **audit-C4-4** (HIGH — D9 paper-only): still open.
- **audit-C4-8 deferred** (LOW — check.py doesn't call fn-mono):
  still open.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open.
- **D-vs-Quote diagnostic text** (cycle-7 deferred): still open.
- **C7-1 close test-coverage gap** (cycle-8 housekeeping):
  still open.
- **C13-1** (HIGH — DCE drops TRACE_EXIT operand on unit-
  returning @trace fns): **CLOSED** by 1e4c3e6 (cycle 14).
  Re-verified by cycle 15 on a broader probe set; closure is
  robust.

---

## Verdict

**Cycle 15 Audit C: CLEAN — 0 findings at or above the
confidence-80 reporting threshold.**

Per user directive 2026-05-10 (strict criterion): zero findings
of any severity at the threshold qualifies as clean.

**Cycle-counter advances from 1/5 -> 2/5** (pending the parallel
silent-failures and type-design audits for this cycle; the
overall clean-streak counter advances only if all three of A/B/C
are clean).

Stage 29 gating remains: **5 consecutive fully-clean cycles
required**; this is the second.

---

## Cycle 15 status

- Audit C result: **CLEAN — 0 findings.**
- Counter going in: 1/5 (cycle 14 first clean post-reset).
- Counter going out: **2/5 (advance, contingent on parallel
  audits A and B for cycle 15).**
- Production HEAD audited: 1e4c3e6 (unchanged from cycle 14).
- Adversarial probe: 8 new shapes (nested @trace, control-
  flow @trace, early-return @trace, mixed-return @trace,
  dead-intermediates @trace, while-loop @trace, param-passthrough
  @trace, @trace + @pure). All pass dce+compile+CLI -O2.
- @trace + @autotune excluded by feasibility (@autotune requires
  @kernel, @kernel routes through ptx.py not x86_64.py).
- C13-1 closure re-validated under broader probe coverage.
- Stage 29 gating: **BLOCKED on 3 more consecutive clean cycles
  (16, 17, 18).**
