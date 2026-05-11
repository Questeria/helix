# Stage 28.8 Cycle 21 — Code-Review Audit (Audit C)

**Date**: 2026-05-11
**Audit HEAD**: `bee36e6` — "Audit 28.8 cycle 21 fix-sweep: close
C20-1 (HIGH, PTX backend isize/usize silent 32-bit)".
**Reviewer**: code-review lens (third audit pass of cycle 21).
**Strict criterion**: cycle counts CLEAN only when **zero findings ≥
80% confidence** of ANY severity.

---

## Scope

The cycle-21 fix-sweep extends four width-keyed PTX-backend tables
to include `isize`/`usize` entries (canon: 64-bit), matching the
typecheck canon + cycle-19 x86_64 classifier fix + cycle-20
const_fold fix. The fix lands in two production-code blocks of
`helixc/backend/ptx.py` (+15/-2) and one new regression test in
`helixc/tests/test_ptx.py` (+28).

The four sites named in the commit message:

1. `_ptx_type_str` (ptx.py:166-173): `.get(ty.name, ".b32")` → now
   has explicit `"isize": ".b64", "usize": ".b64"` entries.
2. `_DTYPE_SIZE` (ptx.py:340-342): class-level dict; now has
   `"isize": 8, "usize": 8`.
3. `_DTYPE_PTX_LOAD` (ptx.py:343-347): class-level dict; now has
   `"isize": "s64", "usize": "u64"`.
4. `_ld_reg_prefix` (ptx.py:355-363): tuple `("i64", "u64")` →
   `("i64", "u64", "isize", "usize")`.

The new test `test_c20_1_isize_usize_treated_as_64_bit_in_ptx`
(test_ptx.py:237-262) pins all four tables.

---

## Method

1. **`git show bee36e6 --stat`** to bound the audit surface:
   - `helixc/backend/ptx.py`: +20 / -3 (mostly audit-stamp comments)
   - `helixc/tests/test_ptx.py`: +28 (new regression test)
   - 3 cycle-19/20 audit docs added (already vetted in prior cycles)
2. **Read the diff** in full for production-code blocks and verify
   the four sites against the cycle-20 silent-failures doc's
   "Site 1: PTX backend dtype maps" analysis (which pre-identified
   the three table fixes; the cycle-21 commit added a fourth site,
   `_ld_reg_prefix`, that the cycle-20 doc had not flagged as a
   separate site — re-classified in cycle-20 Audit B).
3. **Read the current production code** at `ptx.py:140-368` to
   verify the fix is in-place at HEAD, in context of the kernel
   `_format_param` (`.b64` hard-code) and the HBM-tile-indexed
   `_dtype_size` / `_ptx_load_suffix` callers.
4. **Grep `helixc/backend/ptx.py` for `.get(`** — enumerate every
   `dict.get()` call in the file to confirm no fifth silent-fallback
   width table was missed.
5. **Adversarial end-to-end**: attempt to compile
   `@kernel fn k(a: tile<isize, [16], HBM>) {}` through
   parse → typecheck → lower and observe whether the
   NotImplementedError gate at `lower_ast.py:511` fires.
6. **Run `pytest helixc/tests/test_ptx.py`** — confirm the 23 PTX
   tests pass (was 22 → +1 new).
7. **Run the full `pytest helixc/tests/`** to verify no other test
   was disturbed by the four-table extension. (One failure
   surfaced — `test_bootstrap_kovc_full_pipeline_arithmetic` — but
   it is pre-existing, verified by checkout to the parent commit
   `5a1e406` where the same test fails with the same `132` drift.
   Not introduced by cycle 21.)

Read-only audit (no edits to production code or tests).

---

## Verification of the four-site fix

### Site 1 — `_ptx_type_str` (ptx.py:166-173)

Pre-fix:

```python
mapping = {
    "i8": ".b8", "i16": ".b16", "i32": ".b32", "i64": ".b64",
    "u8": ".b8", "u16": ".b16", "u32": ".b32", "u64": ".b64",
    "bool": ".pred",
    "f16": ".f16", "bf16": ".bf16", "f32": ".f32", "f64": ".f64",
}
return mapping.get(ty.name, ".b32")
```

Post-fix:

```python
mapping = {
    "i8": ".b8", "i16": ".b16", "i32": ".b32", "i64": ".b64",
    "u8": ".b8", "u16": ".b16", "u32": ".b32", "u64": ".b64",
    "isize": ".b64", "usize": ".b64",
    "bool": ".pred",
    ...
}
return mapping.get(ty.name, ".b32")
```

**Walk**:

- Type system: `isize`/`usize` are canon-aliased to `i64`/`u64` by
  `typecheck.py:225-228`. PTX `.b{bits}` is a width-tagged storage
  marker (signedness is on the instruction, not the type). So
  `isize`/`usize` → `.b64` is the only correct mapping; matches
  `i64`/`u64`. **Correct.**
- The `.get(..., ".b32")` default is unchanged. It still narrows
  truly unknown names. As cycle 20's silent-failure doc observed,
  this default's only consumer (`emit_device_func` return-type slot
  at line 143) currently emits a stub `ret;` body — no integer
  result actually computes — so the residual `.b32` default is a
  latent-hazard for *new* unknown scalar types but not currently a
  silent-failure window. Not a finding.

### Site 2 — `_DTYPE_SIZE` (ptx.py:340-342)

Post-fix:

```python
_DTYPE_SIZE = {"i8": 1, "u8": 1, "i16": 2, "u16": 2, "f16": 2, "bf16": 2,
                "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8, "f64": 8,
                "isize": 8, "usize": 8}
```

Consumed by `_dtype_size` at ptx.py:349-350:

```python
def _dtype_size(self, dtype: str) -> int:
    return self._DTYPE_SIZE.get(dtype, 4)
```

**Walk**:

- `_dtype_size` is called at ptx.py:302 and 327 inside the
  `TILE_INDEX_LOAD_HBM` / `TILE_INDEX_STORE_HBM` op handlers, used
  as the byte-stride multiplier in `mul.wide.s32 {off}, {idx_reg},
  {self._dtype_size(dtype)}`. For an isize element, the post-fix
  stride is `8` (correct), pre-fix it would have been `4` (silent
  half-stride, every-other-element access).
- The `dtype` string flows from `op.attrs.get("dtype", "f32")` (line
  288/313), populated by `lower_ast.py:2032`-ish from the canon
  dtype name. So `"isize"` as a string is the exact key the table
  must accept; the entry is present. **Correct.**

### Site 3 — `_DTYPE_PTX_LOAD` (ptx.py:343-347)

Post-fix:

```python
_DTYPE_PTX_LOAD = {"i8": "s8", "u8": "u8", "i16": "s16", "u16": "u16",
                    "f16": "f16", "bf16": "bf16",
                    "i32": "s32", "u32": "u32", "f32": "f32",
                    "i64": "s64", "u64": "u64", "f64": "f64",
                    "isize": "s64", "usize": "u64"}
```

**Walk**:

- Signedness pattern: signed types map to `s{bits}` (`i8→s8`,
  `i16→s16`, `i32→s32`, `i64→s64`); unsigned types map to
  `u{bits}` (`u8→u8`, ..., `u64→u64`). Isize is signed in Helix
  (typecheck.py:1816 — range `(-(1<<63), (1<<63)-1)`), so
  `"isize": "s64"` is correct. Usize is unsigned, so
  `"usize": "u64"` is correct.
- Used in `ld.global.<suffix>` / `st.global.<suffix>`
  (ptx.py:306, 329). Pre-fix it would have emitted
  `ld.global.u32` for an isize element (silent truncation +
  sign confusion). Post-fix emits `ld.global.s64`. **Correct.**
- The `_ptx_load_suffix` accessor at line 352-353 returns
  `self._DTYPE_PTX_LOAD.get(dtype, "u32")` — `.get(..., "u32")`
  default is unchanged. Same residual-latent-hazard observation
  as Site 1; not a finding.

### Site 4 — `_ld_reg_prefix` (ptx.py:355-363)

Pre-fix:

```python
if dtype in ("f16", "bf16", "f32", "f64"):
    return "f"
if dtype in ("i64", "u64"):
    return "rd"
return "r"
```

Post-fix:

```python
if dtype in ("f16", "bf16", "f32", "f64"):
    return "f"
if dtype in ("i64", "u64", "isize", "usize"):
    return "rd"
return "r"
```

**Walk**:

- Register pool selection: `%f` for float, `%rd` for 64-bit int,
  `%r` for 32-bit int. The `%rd` pool is declared at ptx.py:130
  as `.reg .b64 %rd<N>`; the `%r` pool at line 129 as
  `.reg .b32 %r<N>`. Pre-fix, isize would have selected `%r`
  (32-bit), then `ld.global.u32` (Site 3 pre-fix) would load
  32 bits into it. Post-fix selects `%rd` (64-bit) and
  `ld.global.s64` loads all 64 bits. **Correct.**
- This is the site the cycle-20 audit doc had *not* enumerated
  (Site 1 in the cycle-20 doc covered only three tables); the
  cycle-21 commit message calls out the cycle-20 Audit-B
  re-classification. The fix-sweep correctly closed all four
  by extending both the dict-based tables and the tuple-based
  membership test. **Lock-step consistent.**

---

## Cross-table consistency check

For an isize element flowing through `TILE_INDEX_LOAD_HBM`:

| Step                | Pre-fix value     | Post-fix value     | Comment            |
|---------------------|-------------------|--------------------|--------------------|
| `_dtype_size`       | 4 (silent half)   | 8 (correct)        | byte stride        |
| `_ptx_load_suffix`  | "u32" (silent)    | "s64" (correct)    | ld.global suffix   |
| `_ld_reg_prefix`    | "r" (32-bit pool) | "rd" (64-bit pool) | register family    |
| `_ptx_type_str`     | ".b32" (silent)   | ".b64" (correct)   | type tag           |

All four post-fix values are mutually consistent: the load reads
8 bytes via `ld.global.s64` into a `%rd` 64-bit register, with the
address advanced 8 bytes per index. No mismatch (e.g., 8-byte load
into a 32-bit register, or 4-byte address-stride with 8-byte load)
is possible after the fix.

The fix-sweep is **lock-step blast-radius-complete** for the four
sites it claims to close.

---

## Adversarial end-to-end probe

Per the cycle-21 prompt: try to emit PTX for
`@kernel fn k(a: tile<isize, [16], HBM>) {}` and observe whether
the NotImplementedError gate at `lower_ast.py:511` fires.

```
typecheck OK
NotImplementedError raised as expected:
  Stage 16 HBM tile param dtype must be f32/i32/f16/bf16; got
  TyName(span=Span(line=2, col=22), name='isize')
```

Confirmed. The typecheck phase accepts the program (`isize` is a
valid scalar), but `lower_ast.py:511`'s allowlist raises before
the dtype string can flow into the PTX `_dtype_size` /
`_ptx_load_suffix` / `_ld_reg_prefix` accessors.

So the cycle-21 fix is **defense-in-depth**: the silent-narrow
window the cycle-21 commit message describes ("reachable from
typecheck-accepted `tensor<isize, ...>` user code via
--emit-ptx") is **not currently reachable through the HBM-tile
path** at HEAD, because the dtype allowlist gate at line 511
intercepts isize/usize. The fix is still correct (closes the
window pre-emptively, before the gate widens), but the cycle-20
Audit-B "reachable at HEAD" characterization is **overstated**
for the HBM-tile path specifically. (The `_ptx_type_str` site
also feeds `emit_device_func`'s return-type slot at line 143,
which is a non-kernel `.func` declaration with a stub `ret;`
body — also not a live silent-failure path because the body
emits no value.)

This is **not a code-review finding** because the fix is
correct, well-stamped, and defense-in-depth is the right
posture for parallel-backend invariants. Recording the
reachability nuance here so the cycle-21 audit ledger doesn't
later mis-cite cycle-20-Audit-B's reachability claim.

---

## Regression test review (test_ptx.py:237-262)

```python
def test_c20_1_isize_usize_treated_as_64_bit_in_ptx():
    """Audit 28.8 cycle 21 C20-1 (HIGH): PTX backend width-keyed
    tables must treat isize/usize as 64-bit, matching typecheck.py
    canon. ..."""
    from helixc.backend.ptx import PtxEmitter
    from helixc.ir import tir
    # Probe class-level tables directly.
    assert PtxEmitter._DTYPE_SIZE["isize"] == 8
    assert PtxEmitter._DTYPE_SIZE["usize"] == 8
    assert PtxEmitter._DTYPE_SIZE["i64"] == 8
    assert PtxEmitter._DTYPE_PTX_LOAD["isize"] == "s64"
    assert PtxEmitter._DTYPE_PTX_LOAD["usize"] == "u64"
    # _ptx_type_str via instance.
    em = PtxEmitter.__new__(PtxEmitter)
    isize_ty = tir.TIRScalar(name="isize")
    usize_ty = tir.TIRScalar(name="usize")
    assert em._ptx_type_str(isize_ty) == ".b64"
    assert em._ptx_type_str(usize_ty) == ".b64"
    # _ld_reg_prefix — isize/usize should pick the 64-bit `rd` pool.
    assert em._ld_reg_prefix("isize") == "rd"
    assert em._ld_reg_prefix("usize") == "rd"
    assert em._ld_reg_prefix("i64") == "rd"
    assert em._ld_reg_prefix("i32") == "r"
```

**Walk**:

- **Coverage**: Pins all four width-keyed sites called out in the
  commit message. Each site has at least one isize and one usize
  assertion. `_ld_reg_prefix("i32") == "r"` and
  `_ld_reg_prefix("i64") == "rd"` are kept as anchor assertions
  to catch regressions where a refactor accidentally collapses
  the 32/64 split.
- **Failure-mode quality**: Each assertion is direct against the
  class-level table or the helper method. Failure pinpoints
  exactly which of the four sites drifted. No pipeline wrapping
  to obscure the failure surface.
- **`PtxEmitter.__new__(PtxEmitter)` for `_ptx_type_str` /
  `_ld_reg_prefix`**: The two methods don't reference `self`
  state, so bare-instance probing is legitimate. The comment
  "bare instance (no __init__ side-effects)" documents the
  intent. Slightly unusual idiom but acceptable for a regression
  test that explicitly avoids running `__init__`.
- **Match between assertions and post-fix tables**: All four
  tables' post-fix entries are pinned (isize→8 byte, isize→s64
  PTX-load, isize→.b64 PTX-type-str, isize→rd reg-prefix; same
  for usize except `_DTYPE_PTX_LOAD["usize"] == "u64"`). All
  pass at HEAD (verified by `pytest -k test_c20_1` — 23/23 PTX
  tests pass).

**Verdict on the regression test**: **clean.** Pins all four
sites, isolates each, fails-fast at the exact accessor.

**Minor note**: the test does not exercise a full --emit-ptx
end-to-end run that would also catch silent-narrow regressions
in *downstream* consumers of the four tables (e.g., the kernel
attribute path or a future signed-vs-unsigned bug in the
`mul.wide.s32` line at ptx.py:302). The cycle-19 forward note
F-20-2 about a full-pipeline regression test still applies to
PTX as well as to const_fold. **Not a current finding** — the
contract-level test pins the specific defect class; a full-
pipeline test would be a stronger guarantee against future
downstream drift, but no production-reachable silent-failure
window currently exists.

---

## Scan for missed `.get(...)` width tables

Grepped `helixc/backend/ptx.py` for `.get(`. All 23 hits are
either:

- Register-map lookups (`self.reg_map.get(...)`) — not width
  tables.
- Attribute lookups on `op.attrs.get(...)` — not width tables.
- The three width-table accessors already enumerated:
  - `mapping.get(ty.name, ".b32")` (line 173, Site 1)
  - `self._DTYPE_SIZE.get(dtype, 4)` (line 350, Site 2)
  - `self._DTYPE_PTX_LOAD.get(dtype, "u32")` (line 353, Site 3)
- `cmp_map.get(cmp_op, "eq")` (line 253) — compare-op suffix
  table for `setp.<cmp>.s32`; not a width table.
- `self.next_reg_by_prefix.get(prefix, 0)` (line 72) — register
  counter; not a width table.

No fifth width-keyed `.get(...)` site was overlooked. The
fix-sweep is complete with respect to the `dict.get(default)`
silent-fallback pattern for the PTX backend.

---

## Other code-quality observations

### Audit-stamp comments

The three stamped comments (ptx.py:159-165, 335-339, 357-358)
follow the established convention:
- Cycle/finding identifier ("Audit 28.8 cycle 21 C20-1 (HIGH)")
- Cross-reference to the canon (typecheck.py / x86_64.py /
  const_fold.py)
- Behavioural description of the pre-fix bug
- Defect-class linkage ("Same defect class as C13-1/C16-1/
  C18-1/C19-1")

This matches the format established in cycles 13/16/18/19/20
(verified by inspection of the cycle-19 fix at x86_64.py:1005-
1017 and the cycle-20 fix at const_fold.py:43-56). **Clean.**

### Signedness asymmetry between `_DTYPE_PTX_LOAD` keys

`_DTYPE_PTX_LOAD["isize"] == "s64"` (signed); `"usize" == "u64"`
(unsigned). This matches the typecheck signedness contract
(typecheck.py:1816 declares isize as signed, usize as unsigned),
and the pattern `i64→s64, u64→u64` for the i/u prefix family
above. The new entries are pattern-consistent. **Correct.**

### Test naming convention

`test_c20_1_isize_usize_treated_as_64_bit_in_ptx` follows the
established convention (cycle/finding identifier in the test
name) — same shape as
`test_c19_1_isize_usize_are_64_bit_in_wrap` from cycle 20 and
the cycle-18/16/13 regression tests. **Clean.**

### Pre-existing test failure unrelated to cycle 21

A full `pytest helixc/tests/` run surfaces one failure:
`test_bootstrap_kovc_full_pipeline_arithmetic` fails with
`compile_and_exec("100 - 50 - 8") == 132` (expected 42). This
test exercises the **bootstrap kovc** binary (not the Python
parser), and the same failure reproduces at the parent commit
`5a1e406` (verified by checkout). **Pre-existing, not introduced
by cycle 21.** Confirmed via:

```
git checkout 5a1e406 -- .
pytest helixc/tests/test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic
# → FAILED (same 132 drift)
git checkout bee36e6 -- .
```

The PTX-targeted 23 tests in `test_ptx.py` all pass. The
cycle-21 fix-sweep is not implicated in this bootstrap
regression. Recorded here for ledger transparency; not a
cycle-21 finding because the failure pre-exists the audited
commit. It is, however, worth a separate-cycle investigation —
flagging this as a forward observation for the next
silent-failures pass (the cycle-19 user-directed
"helixc-python-parser-quirks" memory note is the closest
existing analog; this looks like a parallel-but-distinct bug
in the **bootstrap kovc parser** rather than the Python parser).

### Strict-clean-counter expectation

The cycle-21 commit message states "Clean cycle counter: stays
at 0/5 (cycle 20 NOT clean due to C20-1)". This matches the
strict criterion: cycle 20 surfaced C20-1, which by the strict
rule resets the counter. Cycle 21 (cycles 21-A, 21-B, 21-C) is
the **first** cycle of a fresh streak. If this code-review
audit (21-C) finds zero issues at ≥80% confidence, then cycle
21's three lenses are mutually clean and the counter advances
to **1/5**.

---

## Findings

**None at ≥80% confidence.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

The four-site fix is correct, blast-radius-complete, well-stamped,
and regression-tested. The accompanying test pins all four sites
with direct assertions. The full PTX test suite passes (23/23).
Adversarial end-to-end probe confirms the
`lower_ast.py:511` NotImplementedError gate fires, making the
cycle-21 fix defense-in-depth rather than the only-current-
window-closer.

---

## Forward notes (not findings, per strict re-flag rule)

- **F-21-1 / standing item** — The "centralize the
  `_scalar_width_bits` predicate" Stage-29 refactor remains the
  canonical resolution for this defect class. The cycle-21
  commit message explicitly identifies six HIGH findings in a
  row of this class (C13/C16/C18-B/C18-C/C19/C20) — the
  centralizer is overdue. This forward note has been carried
  since cycle 17; restating here for ledger continuity. Not
  a finding.

- **F-21-2 / Stage-29 territory** — Full-pipeline regression test
  for isize/usize through the PTX pipeline. The cycle-21
  regression test is contract-level (asserts the four tables'
  values directly). A complementary `--emit-ptx` round-trip
  test for `tile<isize, [16], HBM>` would catch downstream
  silent-narrows introduced by new consumers of the four
  tables. This is gated by widening the
  `lower_ast.py:511` allowlist to accept isize/usize HBM
  dtypes; until then, the round-trip test would fail at the
  allowlist gate. Recommended for Stage-29 alongside the
  centralizer landing. Not a current finding (no
  production-reachable silent-failure window open).

- **F-21-3 / parallel-finding observation, NOT a code-review
  finding** — `test_bootstrap_kovc_full_pipeline_arithmetic`
  fails at HEAD and at parent `5a1e406` with the same drift
  (`100 - 50 - 8 → 132`). Pre-exists cycle 21. Looks like a
  bootstrap-kovc parser/codegen bug (right-assoc subtraction
  semantics or a stale binary cache). Suggest a separate audit
  cycle targeted at the bootstrap. Not a cycle-21 finding
  because (a) it's a pre-existing failure and (b) it's
  orthogonal to the PTX backend.

---

## Cross-lens corroboration

Cycle 21's three lenses (Audit A silent-failures, Audit B
type-design, Audit C code-review) target the same fix-sweep at
`bee36e6` from three independent methodologies. Audit C
(this doc) is the third and final pass.

If A and B both return clean and C (this doc) returns clean,
cycle 21 advances the strict-clean-cycle counter from
**0/5 → 1/5**. Four more consecutive clean cycles required to
fire the Stage-29 gate per the user directive 2026-05-10.

---

## Verdict

**Cycle 21 code-review audit: CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

The four PTX width tables are now consistently 64-bit for
isize/usize. The regression test pins all four sites. The fix
is blast-radius-complete, well-stamped, and consistent with the
established audit-stamp comment convention. The adversarial
end-to-end probe confirms the `lower_ast.py:511` gate continues
to fire — the fix is defense-in-depth.

**Clean-cycle counter**: assuming Audits A and B also clean
for cycle 21, advances from **0/5 → 1/5**.

---

## Files touched by this audit

None — read-only audit. Forward notes F-21-1, F-21-2, F-21-3
recorded in this doc only.

## Cross-reference

- Cycle-20 silent-failures (forward note F-20-1 — pre-identified
  three of the four PTX sites):
  `docs/audit-stage28-8-cycle20-silent-failures.md:244-359`
- Cycle-21 fix-sweep commit: `bee36e6`
- Files touched by cycle-21 fix-sweep:
  - `helixc/backend/ptx.py:159-173, 335-347, 355-363`
  - `helixc/tests/test_ptx.py:237-262`
- Canonical width contract (must agree across all backends):
  - `helixc/frontend/typecheck.py:225-228` (`_widen_canon_name`
    aliases isize→i64, usize→u64)
  - `helixc/frontend/typecheck.py:1816` (isize signed-i64 range)
  - `helixc/backend/x86_64.py:1005-1017` (cycle-19 classifier
    fix)
  - `helixc/ir/passes/const_fold.py:43-56` (cycle-20 wrap-table
    fix)
  - `helixc/backend/ptx.py:159-173, 335-347, 355-363` (cycle-21
    PTX-table fix)
- Pre-existing bootstrap-kovc failure (not cycle-21-introduced):
  `helixc/tests/test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic`
