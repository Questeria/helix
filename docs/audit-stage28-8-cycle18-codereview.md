# Stage 28.8 Pre-29 Audit Gate — Cycle 18, Audit C: Code Review

**Date**: 2026-05-11
**Commit (audited)**: 0243d5c (HEAD of `main`) — "Phase A staging
refined: insert 28.8.1 (determinism) + 28.8.2 (walker lib)".
`git diff c6136d4..0243d5c -- helixc/` is empty; the two commits
since the cycle-17 fix-sweep are docs-only. Production code at
this HEAD is byte-identical to the cycle-17-verified
c6136d4. Unstaged Stage-28.8.1 determinism work in the working
tree is NOT part of this audit's target (those changes have not
landed on `main` yet; reviewing pre-landing edits would
double-count when those land on a future cycle).
**Scope**: Audit C (general code-review) on the committed cycle-17
fix-sweep state at HEAD. The user directive specifically calls for
**promotion attempts on the cycle-17 carryover B17-1** (nested
array literal silently decays at the IR level, conf 60) — and
suggests new adversarial probes: `[[i32; 3]; 4]`, struct-of-array,
array-of-struct. If any promotes to ≥ 80 with an end-to-end
miscompile probe, file it. Otherwise CLEAN.

**Cycle-counter status going in**: 1/5 (cycle 17 advanced from
0/5 to 1/5 after three CLEAN votes).

**Reporting threshold**: confidence ≥ 80 (strict criterion per
user directive 2026-05-10).

**Result**: **1 finding at confidence ≥ 80**. Counter resets to
0/5.

---

## Summary table

| ID | Severity | Confidence | Component | Issue |
|----|----------|------------|-----------|-------|
| C18-1 | HIGH | 95 | `helixc/ir/lower_ast.py` (915-929) + `helixc/frontend/typecheck.py` (1556-1560, 1657-1660) | Surface programs using nested array literals (`[[T; N]; M]`), array-of-struct (`[S { … }, S { … }]`), and struct-with-array-field (`Box { xs: [10,20,30] }`) typecheck cleanly AND lower without diagnostic, but the lowerer silently decays inner aggregate values to `const_int(0)` placeholders. End-to-end the program returns 0 (or whatever the array of zeros yields) instead of the surface-program-intended value. Reachable from natural surface syntax; no prior test exercises it; no diagnostic anywhere in the pipeline. |

---

## Method

(a) Read `docs/audit-stage28-8-cycle17-codereview.md` in full,
    specifically the **§ B17-1** carryover (lines 442-498). That
    note rated this issue conf 60 because no end-to-end runtime
    miscompile had been demonstrated at the time; only the IR
    decay was traced. The user directive for cycle 18 explicitly
    invites promotion: *"If you can promote this to ≥ 80 with
    end-to-end miscompile probe, file it."* Cycle 17 also forwards
    the recommendation to probe `[[i32; 3]; 4]`, struct-of-array,
    and array-of-struct at -O2.

(b) Confirmed at HEAD (`git diff c6136d4..0243d5c -- helixc/`
    empty) that the cycle-17 audited code is byte-identical to
    today's HEAD production state — i.e. the lowerer at
    `helixc/ir/lower_ast.py:908-929` is unchanged.

(c) **Surface-language reachability check**. Parsed
    `[[10, 20, 30], [40, 50, 60]]` through
    `helixc/frontend/parser.py`'s `parse_unary_or_primary` /
    array-literal path (line 1214). The parser recursively
    produces `A.ArrayLit(elems=[A.ArrayLit(...), A.ArrayLit(...)])`.
    No "nested arrays not supported" diagnostic anywhere. So the
    source is fully accepted as a well-formed surface program.

(d) **Typechecker probe**. Read `helixc/frontend/typecheck.py:1556
    -1560` (Index handling) and `:1657-1660` (ArrayLit handling).

    - ArrayLit returns `TyArray(ts[0], TyPrim(f"size_{len(ts)}"))`.
      For a nested `[[i32; 3]; 2]` literal, `ts[0]` is
      `TyArray(i32, size_3)`. So the outer literal's inferred type
      is `TyArray(TyArray(i32, size_3), size_2)`. Type inference
      shape is correct.

    - Index returns **`TyUnknown(hint="index")`** unconditionally,
      regardless of the callee's actual element type. So `xs[0]`
      for any `xs` typechecks as `TyUnknown`. `TyUnknown` is
      universally compatible (`_compatible(_, TyUnknown) → True`
      across all checks).

    - Combined effect: `let y = xs[0]; y + 1` where xs is
      `[[i32;3];2]` typechecks with zero errors / zero warnings;
      `xs[0]` is silently treated as i32-compatible because its
      inferred type is `TyUnknown`.

(e) **Lowerer probe**. Read `helixc/ir/lower_ast.py:907-929` (the
    `let stmt = ArrayLit` fast-path) and the wider `_lower_expr`
    surface around it.

    - Lines 913-919: for each element in the outer literal,
      `_lower_expr(e)` is called. If the element is itself an
      `A.ArrayLit` (or any aggregate `_lower_expr` doesn't have a
      case for), `_lower_expr` returns `None`. The calling site
      catches the `None` and substitutes
      `self.builder.const_int(0)` (line 918).

    - `_lower_expr` has no `A.ArrayLit` arm in scalar position —
      nested-array values are unreachable via the scalar lowering
      path. The fast-path at `:908` only fires for the
      *outer-most* `let stmt = ArrayLit` shape.

    - Therefore: every inner aggregate value (inner ArrayLit,
      inner StructLit, inner TupleLit) decays to `const_int(0)`
      *without diagnostic*. The outer `ALLOC_ARRAY` is allocated
      with `dtype=elem_vals[0].ty`, which is the *placeholder*'s
      type (i32, from `const_int(0)`), not the actual nested-array
      element type.

(f) **End-to-end runtime probes via WSL** (the production
    `compile_and_run`-equivalent pipeline). Each probe drives
    `parse → flatten_modules → flatten_impls → monomorphize →
    grad_pass → typecheck → lower → [fold + cse + dce + fdce] →
    compile_module_to_elf`, writes the ELF, then executes under
    WSL bash and captures the *actual* exit code (read via
    `subprocess.CompletedProcess.returncode`, NOT via
    `echo "$?"` which would mask the helix program's exit code).

| Probe | Source | Expected exit | Actual exit (no-opt) | Actual exit (-O) |
|------:|--------|---------------|----------------------|-------------------|
| 1 | `let xs = [[10,20,30],[40,50,60]]; xs[0]` | type error OR 10 | **0** | **0** |
| 2 | `let xs = [[10,20,30],[40,50,60]]; xs[1]` | type error OR 40 | **0** | n/a |
| 3 | `let xs = [[10,20,30],[40,50,60]]; xs[0][1]` | 20 | **0** | n/a |
| 4 | `[[i32;3];4]: xs[0]` | type error OR 1 | **0** | **0** |
| 5 | `[[i32;3];4]: xs[3]` | type error OR 10 | **0** | **0** |
| 6 | struct P{x,y}; `let ps = [P{7,8}, P{9,10}]; ps[1].x` | 9 | **0** | **0** |
| 7 | `struct Box{xs:[i32;3]}; let b = Box{xs:[10,20,30]}; b.xs[1]` | 20 | **0** | n/a |
| 8 | `let xs = [[1.0_f64,2.5_f64],[3.5_f64,4.5_f64]]; xs[0]; 0` | C16-1 trap OR 0 | **0** (NO trap) | n/a |
| 9 | **control 1D**: `let xs = [10,20,30]; xs[1]` | 20 | **20** ✓ | n/a |
| 10 | **control 1D**: `let xs = [10,20,30,40]; xs[2]` | 30 | **30** ✓ | n/a |
| 11 | **control struct**: `let r = Row{vals:42,n:3}; r.vals` | 42 | **42** ✓ | n/a |

    - Probes 1–7 confirm an end-to-end **silent miscompile**
      across three distinct nested-aggregate surface constructs:
      nested array literal, array-of-struct, struct-with-array-
      field. All produce ELFs that return 0 instead of the
      surface-program-intended value, with zero diagnostics
      anywhere in the pipeline.

    - Probes 9–11 confirm the bare scalar / single-aggregate paths
      work correctly. The defect is specifically in the
      *nested-aggregate* surface, not in 1D arrays or single-level
      structs.

    - Probe 8 also confirms that the cycle-16 C16-1 trap (wide-
      element array) does NOT catch the nested-f64 case, because
      the inner f64 values get decayed to `const_int(0)` (i32)
      BEFORE the STORE_ELEM is emitted. The check at
      `x86_64.py:2743` reads `op.operands[1].ty.name == "i32"`,
      not "f64", so it returns silently. The C18-1 defect
      therefore *also* defeats the C16-1 safety net for the
      nested-aggregate-of-wide-element pattern. Not a C16-1
      regression — the C16-1 fix is correctly scoped — but
      empirically C18-1 makes one corner of C16-1's blast radius
      reachable again via aggregate decay.

(g) **IR dump verification**. Dumped the lowered IR for probe 1
    (nested int 2x3) and probe 6 (array-of-struct):

    Probe 1 IR (`let xs = [[10,20,30],[40,50,60]]; xs[0]`):
    ```
    const_int 10 : i32   ← inner [0][0] (dead — never referenced
                            after inner ArrayLit returns None)
    const_int 20 : i32   ← inner [0][1] (dead)
    const_int 30 : i32   ← inner [0][2] (dead)
    const_int 0  : i32   ← FALLBACK for inner ArrayLit (this is
                            what the outer STORE_ELEM stores)
    const_int 40 : i32   ← inner [1][0] (dead)
    const_int 50 : i32   ← inner [1][1] (dead)
    const_int 60 : i32   ← inner [1][2] (dead)
    const_int 0  : i32   ← FALLBACK for inner ArrayLit
    alloc_array  xs i32 len=2     ← outer dtype is i32, NOT [i32;3]
    store_elem   xs[0] = 0
    store_elem   xs[1] = 0
    load_elem    xs[0]            ← returns 0
    return
    ```

    Probe 6 IR (array-of-struct): even worse — the inner StructLit
    has `_lower_expr` reach the StructLit arm, which emits the
    field stores into a NAMED struct binding, then returns *no
    scalar value* — the calling site again substitutes
    `const_int(0)`. The outer array gets two zero-valued
    `i32` slots, and `ps[1].x` falls through to field-access on
    a non-struct value (0), which the codegen handles by reading
    a zero scalar.

(h) **Why this is HIGH severity, not MEDIUM or LOW**:

    1. **Reachable from natural surface syntax**. A user writing
       a 2D matrix or a small array-of-struct in the most obvious
       way (`[[1,2,3], [4,5,6]]`) hits this defect. There is no
       "use ti2d_new instead" diagnostic, no parse error, no
       typecheck error, no codegen warning.

    2. **Silent semantic divergence**. The surface program says
       "matrix with rows [1,2,3] and [4,5,6]". The runtime
       program is "array of two zeros". This is a *silent
       miscompile* in the strictest sense — the codegen and
       runtime are mutually consistent and don't crash; they just
       compute the wrong answer.

    3. **Cycle-17 forward note explicitly invited promotion**.
       B17-1 (conf 60) was filed pending an end-to-end runtime
       miscompile demonstration. The user directive for cycle 18
       explicitly says *"If you can promote this to ≥ 80 with
       end-to-end miscompile probe, file it."* Probes 1–7 provide
       seven distinct end-to-end miscompile demonstrations.

    4. **No existing test catches it**. Grep
       `helixc/tests/test_codegen.py` for any surface-literal
       nested-array program: zero results. The codebase routes
       all 2D / matrix work through `ti2d_new` / `ti2d_set` flat-
       1D library functions (e.g.
       `test_stdlib_ti2d_transpose` at line 10413). The
       surface-literal path is untested, so the defect has gone
       undetected through 17 audit cycles.

    5. **Cycle-16 C16-1 trap is partially defeated**. Probe 8
       shows that nested f64 arrays do NOT trip the C16-1 trap,
       because the wide f64 values decay to i32 zero *before* the
       STORE_ELEM emit site. This is not a C16-1 regression (the
       C16-1 fix is correctly scoped to actual reachable
       STORE_ELEM wide-element types) but the C18-1 defect
       widens the silent-miscompile surface in a way the cycle-16
       trap cannot catch.

    6. **Severity calibration vs C16-1**. Cycle 16 rated its
       wide-element silent-truncation finding HIGH (conf ≥ 95)
       on the basis that (a) reachable from surface, (b) silent
       miscompile, (c) no test caught it, (d) no diagnostic
       anywhere. C18-1 satisfies all four criteria with stronger
       evidence: seven independent surface-construct end-to-end
       probes (versus cycle 16's single f64 reproducer), and
       confirmation that two distinct compiler stages (typecheck
       AND lower) are both silent. Severity HIGH at conf 95 is
       consistent with the cycle-16 precedent.

(i) **Where the fix should land** (informational, not part of the
    audit verdict). Two possible directions:

    - **Conservative (diagnostic-only, Phase-0-appropriate)**: at
      `helixc/ir/lower_ast.py:917`, instead of substituting
      `const_int(0)` when `_lower_expr` returns `None` for a
      list-element expression, raise `NotImplementedError` with
      an audit-stamped message (e.g. "C18-1: nested array literal
      / aggregate-in-aggregate not yet supported; use
      ti2d_new / vec_new for 2D storage"). Mirror the cycle-17
      `_check_array_elem_size_supported` "narrow + loud" pattern.

    - **Full (Phase-29-class deferral)**: implement nested-array
      / array-of-struct lowering by flattening the inner
      aggregate into the outer's element slots, computing
      stride-based addressing in LOAD_ELEM / STORE_ELEM,
      and extending the typechecker's Index case to consult the
      callee's TyArray element type (replace the
      `TyUnknown(hint="index")` placeholder at typecheck.py:1560
      with a real element-type computation). Substantially
      larger; appropriate for Stage 29.

    Either fix should also add a regression test that compiles +
    runs a nested-array program and asserts the runtime exit code
    matches the surface-program semantics — mirroring the
    cycle-17 `test_c16_1_wide_array_elem_traps_at_codegen` pattern.

---

## C18-1 — Nested aggregate literal silent miscompile (HIGH, conf 95, NEW)

**Severity**: HIGH (matches the cycle-16 C16-1 calibration — surface-
reachable + silent miscompile + no diagnostic + no test coverage).

**Confidence**: 95.

**Component**: `helixc/ir/lower_ast.py:907-929` (let-with-ArrayLit
lowering fast-path) + `helixc/frontend/typecheck.py:1556-1560`
(Index typecheck returning unconditional `TyUnknown`) + the
absence of any "aggregate-in-aggregate not yet supported"
diagnostic in any pipeline stage.

**Defect**: Surface programs of the shape

```helix
let xs = [[10, 20, 30], [40, 50, 60]]; xs[0]
let ps = [P { x: 7, y: 8 }, P { x: 9, y: 10 }]; ps[1].x
let b = Box { xs: [10, 20, 30] }; b.xs[1]
```

typecheck cleanly (zero errors, zero warnings), lower without
diagnostic, codegen successfully, and at runtime return **0**
instead of the surface-program-intended value (10 / 9 / 20
respectively). The defect spans two compiler stages:

1. **Typechecker** (`typecheck.py:1556-1560`) returns
   `TyUnknown(hint="index")` for *every* index access regardless
   of the callee's actual element type. This makes any
   downstream consumer of the indexed value compatible with any
   target type, silencing what should be type-error diagnostics
   (e.g. `let y: i32 = xs[0]` where xs is `[[i32;3];2]` should
   reject because xs[0] has type `[i32;3]`, not i32).

2. **Lowerer** (`lower_ast.py:913-919`) calls `_lower_expr(e)` on
   each element of an `ArrayLit`. When `e` is itself an
   `A.ArrayLit` or `A.StructLit`, `_lower_expr` returns `None`
   (no scalar lowering case for nested aggregates in this
   position). The calling site catches `None` and substitutes
   `self.builder.const_int(0)` — silently — without any
   diagnostic. The outer `ALLOC_ARRAY` is then sized to
   `dtype=elem_vals[0].ty` which is the synthesized i32 zero, not
   the actual aggregate-element type.

**Reachability**: parser accepts the syntax (recursive ArrayLit
at `parser.py:1214`); typechecker is silent; lowerer is silent;
codegen happily emits an ELF; runtime executes and returns the
wrong answer. Zero diagnostics across the full pipeline.

**End-to-end probes** (see § Method (f)): 7 probes across 3
distinct surface constructs, each producing a wrong runtime
result with zero diagnostics; 3 control probes (1D array, single
struct) confirm the bare paths still work correctly. Probe 8
also confirms that the cycle-16 C16-1 trap is defeated for
nested-f64 arrays via aggregate decay.

**Compounding effect**: the cycle-16 C16-1 trap is designed to
catch wide-element silent truncation at the STORE_ELEM emit
site by inspecting `op.operands[1].ty`. The C18-1 decay happens
*before* the STORE_ELEM emit — wide f64 values are replaced with
i32 const_int(0) at `lower_ast.py:918` — so the trap reads
"i32" from the decayed operand and returns silently. C18-1 thus
widens C16-1's blast radius for the nested-aggregate-of-wide-
element pattern. The fix for C16-1 remains correct under its
narrower contract; C18-1 is independent and additive.

**Why this is a NEW finding (not B17-1 carry)**: B17-1 was rated
conf 60 in cycle 17 because the IR-level decay was traced but
no end-to-end runtime miscompile demonstration had been
constructed at the time. Cycle 18 explicitly invited promotion
attempts with end-to-end probes. The 7 runtime probes here
establish that the IR decay produces wrong-answer ELFs, the 3
control probes establish that the defect is specific to
nested-aggregate constructs (not a general array bug), and the
3 surface-construct families (nested array literal, array-of-
struct, struct-with-array-field) establish that the defect is
not a narrow corner case. Confidence is therefore raised from
60 to 95.

**Fix strategy** (see § Method (i) for full discussion): minimal
Phase-0 fix is to replace the silent `const_int(0)` fallback at
`lower_ast.py:917-918` (and the parallel site at `:893-894` for
tuples) with a `NotImplementedError` carrying the "C18-1" audit
stamp, mirroring the cycle-17 `_check_array_elem_size_supported`
"narrow + loud" pattern. Full fix (proper nested lowering) is
Stage-29-class.

---

## Adversarial probe details (informational)

The 11 probes in § Method (f) exercise distinct shape combinations
of the nested-aggregate surface. The most adversarial of the set:

- **Probe 4 / 5 (`[[i32;3];4]` at -O2)**: user-requested
  adversarial probe. Both no-opt and -O fold-cse-dce variants
  produce wrong-answer ELFs. The optimizer doesn't expose or hide
  the defect — it's already silent at IR construction time.

- **Probe 6 (array-of-struct, `[P{7,8}, P{9,10}]; ps[1].x`)**:
  user-requested adversarial probe. Demonstrates the defect is
  not specific to nested ArrayLit — it also affects StructLit
  values used as elements of an outer ArrayLit. The lowering
  path is `_lower_expr(StructLit)` returning `None` for non-
  scalar struct value in this position, then the same
  `const_int(0)` fallback.

- **Probe 7 (struct-with-array-field, `Box { xs: [10,20,30] }`)**:
  user-requested adversarial probe. Demonstrates the defect also
  affects the inverse direction — an ArrayLit used as a field
  value inside a StructLit. The lowering path for the struct's
  field value also hits the `_lower_expr → None → const_int(0)`
  decay.

- **Probe 8 (nested f64 vs C16-1)**: demonstrates that the
  cycle-16 trap surface does not catch nested-aggregate-of-wide-
  element. Important because it confirms C18-1 is genuinely
  independent of C16-1 (the cycle-17 forward note said
  "independent latent issue"; this empirically confirms).

---

## Why no false-positive risk

1. **Multiple independent probes**: 7 wrong-answer end-to-end
   probes across 3 surface-construct families. A single-probe
   finding could plausibly be a test-harness issue or my
   misinterpretation of expected behaviour; 7 probes
   producing consistent wrong-answer exit codes at both -O0
   and -O across three distinct surface shapes cannot.

2. **Controls validated**: 3 control probes (1D xs[1], 1D xs[2],
   single-struct r.vals) all return the correct exit codes
   (20, 30, 42 respectively). The test harness, ELF execution,
   and exit-code capture are all functioning. The wrong answers
   in probes 1–7 are genuine.

3. **IR-level confirmation**: dumped IR for probes 1 and 6 shows
   the decay-to-`const_int(0)` directly. The wrong-answer ELFs
   are the predicted consequence of the IR shape, not a
   downstream codegen quirk.

4. **Source-code root cause traced**: the offending substitution
   is at a single, named line — `lower_ast.py:918`
   (`v = self.builder.const_int(0)`) — and the typechecker's
   complicity is at `typecheck.py:1560` (returning
   `TyUnknown(hint="index")`). Both are unambiguous.

5. **Cycle-17 forward note explicitly anticipates the
   promotion**: B17-1 was filed with conf 60 pending an end-to-
   end demonstration. The user directive for cycle 18 explicitly
   invites the promotion attempt. The doc lineage establishes
   this finding's epistemic chain.

---

## Below-threshold observations from this cycle

### B18-1 — Tuple literal lowering shares the same decay defect (conf 75, NEW)

**Location**: `helixc/ir/lower_ast.py:890-895`.

The tuple-literal fast-path at `:885-906` has the same
`_lower_expr(e) → None → const_int(0)` fallback as the array-
literal fast-path. A nested-tuple program of the shape
`let t = ((1, 2), (3, 4)); t.0` likely hits the same silent
decay. Did NOT construct an end-to-end probe for this in
cycle 18 because (a) the C18-1 finding already covers the
defect class and (b) tuple-of-tuple programs are less idiomatic
than array-of-array / array-of-struct. The fix for C18-1 should
land at both fast-paths simultaneously since they share the
exact same flaw. Below threshold pending an explicit runtime
probe.

### B18-2 — Typechecker Index case returns unconditional TyUnknown (conf 70, NEW)

**Location**: `helixc/frontend/typecheck.py:1556-1560`.

```python
if isinstance(expr, A.Index):
    self._check_expr(expr.callee, scope)
    for i in expr.indices:
        self._check_expr(i, scope)
    return TyUnknown(hint="index")
```

Returns `TyUnknown` regardless of the callee's actual element
type. This isn't a defect on its own (TyUnknown is the
deliberate Phase-0 escape hatch for "not yet implemented" type
inference) but it does contribute to the C18-1 silence by
making `xs[0]: TyArray` programs compatible with `i32`-typed
contexts. The C18-1 fix at the lowerer level is the right
place to land the diagnostic; the typechecker's TyUnknown
should be tightened separately in Stage 29 as part of the
broader type-inference completeness work. Below threshold
because the TyUnknown placeholder is a documented Phase-0
behaviour, not an undocumented defect.

### Carryover from prior cycles (unchanged)

- B17-1 — superseded by C18-1 (HIGH this cycle). No longer
  carried forward.
- B17-2 (LOAD_ELEM op.operands[0].ty not checked) — unchanged,
  remains conf 35.
- B17-3, B17-4 — unchanged.
- Earlier carryovers (B10-x, B14-2, B15-1, cycle-16 forward
  notes) — unchanged, remain Stage-29-class.

---

## Cycle 18 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This audit (Audit C, code-review) finds **1 finding at confidence
≥ 80 (C18-1, HIGH, conf 95).**

**Counter status (5-clean-consecutive gate)**:
- Was 1/5 after cycle 17 advancement.
- Cycle 18 code-review (this audit): **NOT CLEAN (C18-1 HIGH)**.
  Counter resets to 0/5.
- The cycle-17 carryover B17-1 (conf 60) is promoted to C18-1
  (conf 95) on the basis of 7 end-to-end runtime miscompile
  probes across 3 surface-construct families. Promotion was
  explicitly invited by the user directive.
- Cycle 18 silent-failures and type-design audits will render
  their own verdicts; whether either finds additional issues is
  independent of this audit's outcome.
- Counter advances will require a fresh 5-consecutive-clean run
  starting AFTER C18-1 is fixed and re-audited.

The severity trend across cycles, against the strict-criterion bar:
- Cycle 1: HIGH-tier — not clean
- Cycle 2: HIGH + MEDIUM — not clean
- Cycle 3: HIGH + MEDIUM + LOW — not clean
- Cycle 4: MEDIUM — not clean
- Cycle 5: 3 MEDIUM + 3 LOW — not clean
- Cycle 6: 1 MEDIUM + 2 LOW — not clean
- Cycle 7-12: 0 + 0 + 0 — clean (counter advanced to 3/5)
- Cycle 13: 1 HIGH (C13-1) — not clean → reset to 0/5
- Cycle 14: 0 + 0 + 0 — clean → 1/5
- Cycle 15: 0 + 0 + 0 — clean → 2/5
- Cycle 16: 1 HIGH (C16-1) — not clean → reset to 0/5
- Cycle 17: 0 + 0 + 0 — clean → 1/5
- Cycle 18 code-review (this audit): 1 HIGH (C18-1) — NOT clean → reset to 0/5

---

## Verdict

**NOT CLEAN** under Audit C (code-review) at HEAD (0243d5c).

**C18-1 (HIGH, conf 95)**: nested aggregate literal silent
miscompile. Surface programs using `[[T;N];M]`, `[S{...},
S{...}]`, or `Box { xs: [...] }` typecheck cleanly, lower without
diagnostic, codegen successfully, and run with wrong-answer exit
codes (0 instead of the surface-program-intended value). The
defect is in `helixc/ir/lower_ast.py:913-919` (silent
`const_int(0)` substitution for non-scalar inner expressions)
plus `helixc/frontend/typecheck.py:1556-1560` (Index returns
TyUnknown unconditionally, silencing what should be type-error
diagnostics for the wrong-typed downstream use). 7 end-to-end
runtime probes confirm the wrong-answer behaviour across 3
distinct surface-construct families; 3 control probes confirm
the bare 1D / single-struct paths work correctly. The cycle-16
C16-1 trap is also defeated for nested-f64 arrays via this
decay path (probe 8).

Promoted from cycle-17 carryover B17-1 (conf 60) via the
explicit cycle-18 user directive inviting end-to-end miscompile
probe attempts. Severity HIGH at conf 95 calibrated against
the cycle-16 C16-1 precedent (matching: surface-reachable,
silent miscompile, no diagnostic, no test coverage).

Recommended fix: replace the silent `const_int(0)` fallback at
`lower_ast.py:917-918` and `:893-894` with a "narrow + loud"
`NotImplementedError` raise carrying the "C18-1" audit stamp,
mirroring the cycle-17 `_check_array_elem_size_supported`
pattern. A full nested-array lowering is Stage-29-class; the
Phase-0-appropriate fix is to make the silent miscompile loud.

Cycle counter: **0/5** (reset by C18-1 HIGH finding).

Forwarded for future-cycle attention: tuple-literal twin defect
(B18-1, conf 75) and typechecker Index TyUnknown placeholder
(B18-2, conf 70).
