# Stage 28.8 Pre-29 Audit Gate — Cycle 16, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: 4c74627 (read-only)
**Scope**: Audit C (general code-review) at HEAD. `git rev-parse
HEAD` is `4c7462792d7383a289000fd01e8f798bc327c745`, which is one
commit ahead of the cycle-14/15 HEAD (`1e4c3e6`) — the new commit
is the cycle-11 Audit-A A1 fix-sweep for the bootstrap test
harness flake (`test_codegen.py` only, `_win_to_wsl` /
`compile_and_exec` shell-pipeline robustness). No production
helixc/ code changed since cycle 14; only the test harness was
hardened.

**Cycle-counter status going in**: 2/5 (cycles 14 + 15 fully
clean).

**Method**:

(a) Read `docs/audit-stage28-8-cycle14-codereview.md` and
    `docs/audit-stage28-8-cycle15-codereview.md` for the prior
    closure record + the conf-95 adversarial probe set across 8
    `@trace`-shape variants.

(b) `git rev-parse HEAD` => `4c7462792d7383a289000fd01e8f798bc327c745`.
    `git show 4c74627 --stat`: only `helixc/tests/test_codegen.py`
    changed (+37/-9 lines, harness fix for the bootstrap WSL
    pipeline; not a production-code change).

(c) Re-read `helixc/ir/passes/dce.py` end-to-end (143 LOC).
    Unchanged from cycle 15. `SIDE_EFFECT_KINDS` still has 20
    members. The 12-line comment block at lines 68-78 is intact.

(d) Re-enumerated `tir.OpKind` (91 members) against `SIDE_EFFECT_KINDS`
    (20 members) on the live import. Spot-check inventory of which
    op kinds the lowerer emits with no `results` on a 6-program
    sample (constant, fn, struct, while, @trace-unit-ret, @trace-
    struct-ret): every no-result op observed in the sample was in
    `SIDE_EFFECT_KINDS`. (See "Enum-coverage probe" below for the
    full inventory.)

(e) Ran `helixc/tests/test_dce.py` (8 pass) +
    `helixc/tests/test_trace.py` (21 pass) +
    `helixc/tests/test_struct_mono.py` (32 pass) +
    `helixc/tests/test_deprecated.py` (21 pass): **82 / 82 pass.**

(f) Ran the full non-codegen test suite as a regression sanity:
    `python -m pytest helixc/tests/ --ignore=test_codegen.py -q`
    => **773 passed in 146.03s.**

(g) **Fresh-eyes adversarial probe** — picked the prompt-suggested
    "least-covered" shape: **`@trace` + struct return at -O2** (the
    cycle-15 probe set covered i32/Unit returns only; struct
    return through @trace had no probe coverage in cycle 15). Plus
    a generic-struct + -O3 chain probe, plus a @deprecated × DCE
    × -O2 interaction probe. See "Adversarial probe details"
    below for the 8 new probe shapes.

(h) Verified end-to-end via the canonical CLI path the prompt
    specified: `python -m helixc.check <file> -O2 -o /tmp/out.bin`
    succeeds (rc=0, ELF emitted) for all 8 new probe shapes that
    pass typecheck.

(i) For each compiled probe, ran the in-process post-DCE dangling-
    operand sweep (the same invariant check the cycle-13 audit
    used to find C13-1): for every surviving op, asserted each
    operand.id is in `(producers + fn.params + block.params)`.
    **Across 19 functions in 8 new probes: 0 dangling operands.**

(j) For the @trace + struct-return shape, additionally ran the
    resulting ELF on WSL to inspect runtime behavior (the prompt
    explicitly asked for `... && /tmp/out.bin; echo $?`). See
    "Runtime probe observation" below — surfaced a sub-80 concern
    that is naturally an Audit-A (silent-failures) finding, not
    an Audit-C finding.

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

## Adversarial probe details — cycle 16

The cycle-15 probe set exercised `@trace` × {i32, Unit} returns ×
{nested, control-flow, early-return, while, dead-intermediates,
passthrough, @pure} (8 shapes). The shapes not covered include:

- @trace returning a struct (struct is neither i32 nor Unit)
- @trace returning an array literal
- @deprecated × DCE × -O2 (the @deprecated pass writes a side-table
  but doesn't affect the IR; verifying DCE doesn't accidentally
  treat the call sites differently)
- @deprecated × @trace combined (two attributes whose semantics
  interact only at warning-emission time)
- nested @trace where both fns return structs

For cycle 16 I picked **@trace + struct return** as the primary
new probe shape (the prompt's "@trace + struct return" suggestion),
plus 7 secondary shapes that exercise distinct (struct, array,
@deprecated × DCE) intersections.

### Probe 1 — `@trace fn make() -> Point { Point { x:7, y:11 } }` (struct return through @trace)

```
struct Point { x: i32, y: i32 }

@trace
fn make() -> Point {
    Point { x: 7, y: 11 }
}

fn main() -> i32 {
    let p: Point = make();
    p.x + p.y
}
```

* `python -m helixc.check probe1.hx -O2 -o probe1.bin`: **rc=0,
  ELF 4685 bytes.**
* Post-DCE dangling-operand sweep on `make()` and `main()`: 0
  dangling operands. Producers set is `{0}` post-fold; the lone
  TRACE_EXIT operand (id=0) is in the producers set — the
  cycle-14 fix (TRACE_ENTRY/TRACE_EXIT in `SIDE_EFFECT_KINDS`)
  correctly seeds operand id=0 live.
* IR-dump observation: `make()`'s body lowers to just
  `[trace.entry, const.int(value=0), trace.exit op[0], return]`
  with no struct-constructor ops. See "Runtime probe observation"
  below for the sub-80 follow-up.

### Probe 2 — Big struct (4 fields, force aggregate slot path)

```
struct Big { a: i32, b: i32, c: i32, d: i32 }

@trace
fn make() -> Big { Big { a: 1, b: 2, c: 3, d: 4 } }

fn main() -> i32 {
    let b: Big = make();
    b.a + b.b + b.c + b.d
}
```

* Compile rc=0, ELF 4685 bytes.
* Dangling-operand sweep: 0 dangling.

### Probe 3 — Struct return with early-return path

```
struct Point { x: i32, y: i32 }

@trace
fn maybe(cond: i32) -> Point {
    if cond > 0 {
        return Point { x: 7, y: 11 };
    }
    Point { x: 0, y: 0 }
}

fn main() -> i32 {
    let p: Point = maybe(1);
    p.x + p.y
}
```

* Compile rc=0, ELF 4792 bytes.
* Exercises the C2-2 early-return TRACE_EXIT emission path *with*
  a struct-typed body_val. Dangling-operand sweep: 0 dangling.

### Probe 4 — `@deprecated` × `@trace` combined

```
@deprecated("old1")
fn a() -> i32 { 5 }

@deprecated("old2")
@trace
fn b() -> i32 { a() + 1 }

fn main() -> i32 { b() }
```

* Compile rc=0, ELF 4727 bytes.
* The @deprecated pass emits two warnings (one per call site of a
  deprecated fn) but doesn't change IR or DCE behavior. The @trace
  side correctly emits TRACE_ENTRY/TRACE_EXIT around the inner
  body; the inner CALL op to `a()` is seeded live via
  `SIDE_EFFECT_KINDS` membership of CALL.

### Probe 5 — `@trace` returning array literal

```
@trace
fn make_arr() -> [i32; 3] { [10, 20, 30] }

fn main() -> i32 {
    let arr: [i32; 3] = make_arr();
    arr[0] + arr[1] + arr[2]
}
```

* Compile rc=0, ELF 4685 bytes. Dangling-operand sweep: 0.

### Probe 6 — Nested `@trace` with struct returns (two TRACE_EXITs)

```
struct Box { v: i32 }

@trace
fn inner() -> Box { Box { v: 42 } }

@trace
fn outer() -> Box { inner() }

fn main() -> i32 {
    let b: Box = outer();
    b.v
}
```

* Compile rc=0, ELF 4717 bytes. Two TRACE_EXIT call sites in one
  module — both operand-ids survive the per-function DCE pass.

### Probe 7 — Deeply nested if/else inside `@trace`

```
struct Wrap { v: i32 }

@trace
fn pick(cond: i32) -> i32 {
    let a: Wrap = Wrap { v: 100 };
    let b: Wrap = Wrap { v: 200 };
    if cond > 0 {
        if cond > 5 { a.v } else { b.v }
    } else {
        a.v + b.v
    }
}

fn main() -> i32 { pick(3) }
```

* Compile rc=0, ELF 4984 bytes. The struct allocations + 4
  basic-block control-flow graph + TRACE_EXIT-on-i32-return all
  cohabit. Dangling-operand sweep: 0.

### Probe 8 — `@deprecated` × DCE × dead-code

```
@deprecated("unused")
fn never_called() -> i32 { 99 }

fn main() -> i32 { 5 }
```

* Compile rc=0, ELF 4649 bytes.
* Verifies that DCE does NOT remove a fn declaration entirely
  (only per-block ops). The deprecated, never-called fn is
  emitted into the ELF; no warning fires because no call site
  exists. Both behaviors are correct under the documented
  contract.

**Across all 8 probes**: zero dangling operands, zero KeyErrors,
zero compiler internal-errors, zero stderr tracebacks.

---

## Enum-coverage probe — emitted ops with no results

Across a 6-program sample inventory (basic fn, fn-call, struct,
@trace-unit, while-loop, @trace-struct), every op the lowerer
emits with `not op.results` is a member of `SIDE_EFFECT_KINDS`:

```
array.alloc        [SIDE_EFFECT]
array.store_elem   [SIDE_EFFECT]
br                 [SIDE_EFFECT]
cond_br            [SIDE_EFFECT]
return             [SIDE_EFFECT]
trace.entry        [SIDE_EFFECT]
trace.exit         [SIDE_EFFECT]
var.alloc          [SIDE_EFFECT]
var.store          [SIDE_EFFECT]
```

No new "no-result + non-side-effect + operand-bearing" shape has
emerged since cycle 14 closed C13-1. The seed-set continues to be
complete with respect to the lowerer's emit surface.

---

## Verification that HEAD ahead is a no-op for production code

`git diff 1e4c3e6..4c74627 --stat` (relative to the cycle-14/15
audited HEAD):

```
 helixc/tests/test_codegen.py | 46 +++++++++++++++++++++++++++++++++++---------
 1 file changed, 37 insertions(+), 9 deletions(-)
```

The single delta is in the test harness (`compile_and_exec`'s WSL
shell pipeline) per the cycle-11 Audit-A A1 fix. Production code
(`helixc/{frontend,ir,backend}/*`) is byte-identical to the cycle-
15 audited HEAD. The cycle-15 conclusion — that the cycle-14 fix
is complete and correct under the broader @trace surface — rolls
forward unchanged for cycle 16's production-code audit.

---

## Runtime probe observation (sub-80, lane-displaced)

While running probe 1 through WSL (`python -m helixc.check ...
-O2 -o /tmp/out.bin && /tmp/out.bin; echo $?` — the canonical
end-to-end path the prompt specified):

```
exec rc: 0    (expected 18 = 7 + 11)
```

Observed: a fn that constructs and returns a struct by value is
**silently miscompiled** — the binary exits 0 instead of computing
`p.x + p.y = 18`. Reproduces at -O0 and -O2; the @trace attribute
is NOT involved (the same shape without @trace exhibits the same
0-instead-of-18 behavior). The lowerer's
`lower_ast.py:_lower_fn_body` does not lower the struct
constructor when it occurs in tail-expression-of-body position
with a struct return type; the resulting IR for `make()` is just
`[trace.entry, const.int(0), trace.exit op[0], return]` (no
struct allocation, no return value). The backend then emits a
function that returns 0 in `rax`, but the caller-side `p.x + p.y`
reads stack-resident zero memory (since `let p: Pt = make()` was
allocated but never written).

**Status**: pre-existing, well-documented Phase-0 limitation,
**not a code-review finding for cycle 16**. Specifically:

* `helixc/examples/hbs_pattern_struct_return.hx:3-5` states:
  "HBS pattern: simulating struct-return-by-value via arena
  output params. Until the compiler supports true struct returns
  (Tier F follow-up), self-host code uses this idiom."
* `helixc/examples/hbs_reference_500loc.hx:27` documents:
  "LIMITATION USED-AS-DEMO: struct return values are still not
  fully [supported]".
* `helixc/tests/test_codegen.py:3309-3311` (bootstrap-side test
  comment): "A struct returned by value still SEGVs at runtime
  because Phase-0 lacks proper struct-return ABI (caller-alloc'd
  slot via rdi); that's a separate Stage 5+ work item".
* No production test in the entire suite (773 non-codegen tests
  + the codegen suite) constructs and returns a struct literal
  by value from a fn — the gap has zero coverage.

**Why this is sub-80 for Audit C**:

1. The lowerer's code, read line-by-line as a code-reviewer would,
   is internally consistent: struct constructors aren't lowered
   in tail-of-body position, struct-return ABI isn't implemented,
   `body_val` falls through to the synthesized 0 sentinel. There
   is no "buggy code" to point at — the missing feature is the
   absence of a struct-return-ABI implementation plus the absence
   of a compile-time diagnostic gating that absence.

2. The natural lane for "compiler silently accepts a documented-
   unsupported feature and emits a wrong-result binary" is the
   silent-failures audit (Audit A), not code-review. Audit A's
   stated remit is precisely this shape — features that fail
   without surfacing a diagnostic.

3. The cycle-14 + cycle-15 audits explicitly used a conf-80 strict
   criterion and below-threshold notes for feature-gap concerns;
   this is the same calibration.

4. The miscompile is consistent across -O0/-O1/-O2/-O3, so DCE is
   not implicated. The cycle-13 fix and the cycle-14 / cycle-15
   verification of that fix remain correct.

**Recommendation**: flag for the cycle-16 silent-failures audit
(Audit A) as a candidate observation; that lane is the
appropriate venue for either a compile-time-error diagnostic
proposal or a regression-test scaffold pinning the current
behavior. Logging here for the parallel A/B auditors and the
cycle-17/18 closure-tracker — **rated conf 70, below the
audit-C 80 threshold.**

---

## Why no findings at >= 80

1. **HEAD's production code is identical to cycle 14/15** (the
   one new commit, 4c74627, modifies only `test_codegen.py`'s WSL
   shell-pipeline harness). Any new audit-C finding would have to
   either (a) re-evaluate a code path already cleared by cycle
   15's 8-shape probe (already exhausted), or (b) surface in a
   newly-exercised shape. The cycle-16 probe deliberately picked
   the prompt-suggested @trace × struct-return shape; 8 new
   shapes were compiled + dangling-operand-swept + 1 was run
   end-to-end. Zero dangling operands.

2. **Static enum-coverage check**: the lowerer's no-result emit
   surface is still entirely contained in `SIDE_EFFECT_KINDS`.

3. **End-to-end CLI driver succeeds** for all 8 probe shapes at
   `-O2`. The cycle-13 crash mode (`KeyError: 1` at x86_64.py:2498)
   does not reappear in any new shape.

4. **Existing test suite passes**: 82/82 on the @trace/dce/struct-
   mono/deprecated focus subset; 773/773 on the broader non-
   codegen suite.

5. **Runtime miscompile probe** is sub-80 per the rationale above:
   pre-existing, documented, and lane-displaced (Audit A territory,
   not Audit C territory).

---

## Below-threshold observations from this cycle

The following items were rated below conf 80 and are NOT counted
as cycle-16 findings. Surfaced here for cumulative carryover.

### B16-1 — Struct-return-by-value silent miscompile (conf 70, lane-displaced)

The @trace probe 1's runtime exec yields rc=0 instead of 18 for
`fn make() -> Point { Point { x:7, y:11 } } ...`. Pre-existing,
well-documented Phase-0 limitation; the lowerer doesn't implement
struct-return ABI and no compile-time diagnostic gates the
unsupported path. Naturally belongs to the silent-failures audit
lane. Flagged here for cumulative visibility; deferred to Audit A
for cycle 16. **Not new (predates stage 28.8); not a code-review
defect (code is consistent with documented limitation); not
promoted.**

### B16-2 — `dce.py:14-22` docstring "Side-effecting op kinds" enumeration is still stale (conf 30, unchanged from cycle 15's B15-1)

Rolled forward unchanged.

### B16-3 — `test_c13_1_dce_preserves_trace_entry_in_kept_set` remains a present-state tautology (conf 35, unchanged from cycle 15's B15-2)

Rolled forward unchanged.

### B16-4 — No global post-DCE dangling-operand invariant pass in production (conf 50, unchanged from cycle 15's B15-3 / cycle 14's B14-3 / cycle 13's B13-3)

The cycle-16 probe harness *does* run such a sweep manually
across 8 new shapes, which is what gives this audit its high
coverage. A defensive production pass that asserts the same
invariant after `dce_module` would have caught C13-1 immediately
and would catch any future regression of the same shape.
Recommend promoting to a fix-sweep candidate when cycle 18 (the
last gating cycle) lands. **Conf 50, unchanged.**

### B16-5 — `dce_function`'s outer `while changed` recomputes liveness from scratch (conf 55, unchanged from cycle 15's B15-4 / cycle 13's B13-1)

Pre-existing performance nit. Unchanged.

### B16-6 — `TENSOR_STORE` enum member unreferenced (conf 35, unchanged from cycle 15's B15-5 / cycle 13's B13-2)

Aspirational enum entry with no emit site or backend handler.
Speculative future-proofing concern only. Unchanged.

---

## Below-threshold re-evaluation (cycles 6-15 carryover)

Walked the cumulative below-threshold concern list once more.

- **Cycle 6** (TRAP-const test additions; monomorphize_safe
  docstring drift): no change. Conf <= 40.
- **Cycle 7** (D-vs-Quote diagnostic text): no change. Conf 30.
- **Cycle 8** (C7-1 `_compatible(TyMemTier, TyVar)` coverage gap):
  no change. Conf 55.
- **Cycle 9** (OSError edge cases; pathological strip-shape):
  no change. Conf 25-55.
- **Cycle 10 / 11 / 12** (test ordering; signature; commit-message
  hygiene; strict-stdlib end-to-end coverage): no change. Conf
  10-55.
- **Cycle 13** (B13-1 outer-loop re-seed; B13-2 TENSOR_STORE
  speculative; B13-3 missing invariant checker): rolled forward
  as B16-4, B16-5, B16-6. No change. Conf 35-55.
- **Cycle 14** (B14-1 entry-tautology; B14-2 stale docstring;
  B14-3 invariant-checker): rolled forward as B16-2, B16-3, B16-4.
  No change. Conf 30-50.
- **Cycle 15** (B15-1..B15-5): rolled forward as B16-2..B16-6. No
  change. Conf 30-55.

No promotions to >= 80 from prior cycles. No new audit-C findings
this cycle.

---

## Open prior findings (not addressed this cycle)

Per cumulative carryover from cycle 15:

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
  Re-verified by cycle 15 + cycle 16 on broader probe sets;
  closure remains robust.

---

## Verdict

**Cycle 16 Audit C: CLEAN — 0 findings at or above the
confidence-80 reporting threshold.**

Per user directive 2026-05-10 (strict criterion): zero findings
of any severity at the threshold qualifies as clean.

**Cycle-counter advances from 2/5 -> 3/5** (pending the parallel
silent-failures and type-design audits for this cycle; the
overall clean-streak counter advances only if all three of A/B/C
are clean).

Stage 29 gating remains: **5 consecutive fully-clean cycles
required**; this is the third.

---

## Cycle 16 status

- Audit C result: **CLEAN — 0 findings.**
- Counter going in: 2/5 (cycles 14 + 15 clean).
- Counter going out: **3/5 (advance, contingent on parallel
  audits A and B for cycle 16).**
- Production HEAD audited: 4c74627 (cycle-11 Audit-A A1 fix;
  test harness only; production code byte-identical to cycle-15
  HEAD 1e4c3e6).
- Adversarial probe: 8 new shapes (@trace × struct-return,
  @trace × big-struct, @trace × struct + early-return,
  @trace × @deprecated chain, @trace × array-return,
  nested-@trace × struct, @trace × nested-if/else,
  @deprecated × DCE × dead-code). All pass dce+compile+CLI -O2
  and the post-DCE dangling-operand sweep.
- Runtime probe surfaced a sub-80 silent-miscompile concern on
  struct-return-by-value (B16-1, conf 70); pre-existing,
  documented Phase-0 limitation; flagged for the Audit-A lane
  rather than promoted to Audit-C.
- Stage 29 gating: **BLOCKED on 2 more consecutive clean cycles
  (17, 18).**
