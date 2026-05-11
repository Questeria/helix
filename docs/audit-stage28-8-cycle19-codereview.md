# Stage 28.8 Pre-29 Audit Gate — Cycle 19, Audit C: Code Review

**Date**: 2026-05-11
**Commit (audited)**: 0803902 — "Audit 28.8 cycle 19 fix-sweep: close C18-1
(HIGH, isize/usize silent trunc)"
**Repo HEAD (advisory)**: at audit time, the branch tip is 5a1e406 (7 commits
past 0803902). Of those, e53e510 ("remove stale audit-cycle-18 probe scripts")
removes the two `_audit18_probe*.py` files; 7649c15 / a04036b / 9436810 /
44f17e6 are Stage-28.8.2 walker refactors that wire `helixc/frontend/ast_walker.py`
into `panic_pass` / `deprecated_pass` / `grad_pass` / `struct_mono`; cf9cf7e is
a separate cycle-19+ close of audit-C18-1 (nested aggregate decay); 5a1e406 is
the cycle-20 fix-sweep for C19-1 (const_fold isize drift) — out of scope.
Audit scope per user directive: cycle 19's fix-sweep at 0803902, the cycle-19
counter is fresh at 0/5.

**Scope**: Audit C (general code-review) on the cycle-19 fix-sweep at 0803902,
which closes cycle-18 finding C18-1 (HIGH, conf ≥ 90 per cycle-18 audit B):
`_is_i64_type` / `_is_u64_type` at `helixc/backend/x86_64.py` used strict
name-equal on `i64` / `u64`, missing the pointer-width aliases `isize` /
`usize`. Two reachable silent miscompiles documented:

1. CONST_INT emit at `x86_64.py:1144-1153` — `let x: isize = 5_000_000_000;`
   takes the 32-bit `mov_eax_imm32(value & 0xFFFFFFFF)` branch.
2. Fn-param spill at `x86_64.py:971-989` — `fn f(x: isize)` spills with the
   32-bit `mov [rbp-N], edi` variant, dropping the top 32 bits of RDI.

The fix-sweep extends both classifiers to accept the alias name as well:

```python
def _is_i64_type(self, ty: tir.TIRType) -> bool:
    return isinstance(ty, tir.TIRScalar) and ty.name in ("i64", "isize")

def _is_u64_type(self, ty: tir.TIRType) -> bool:
    return isinstance(ty, tir.TIRScalar) and ty.name in ("u64", "usize")
```

Plus a regression test `test_c18_1_isize_usize_recognized_as_64bit` that pins
the classifier contract directly.

**Cycle-counter status going in**: 0/5 (reset by cycle-18 C18-1 HIGH finding).

**Strict reporting threshold**: confidence ≥ 80 per user directive 2026-05-10.

**Result**: **0 findings at or above the confidence-80 reporting threshold.**

---

## Method

(a) Read `docs/audit-stage28-8-cycle17-codereview.md` for house format (summary
    table, adversarial probe block, below-threshold carryover, why-no-findings
    section). Followed that format here.

(b) Ran `git show 0803902 --stat` and `git show 0803902 -- helixc/backend/x86_64.py
    helixc/tests/test_codegen.py`. The production-code delta is +11 lines in
    `x86_64.py` (split between two 1-line classifier extensions + 5 + 5 lines
    of audit-stamped comments) and +27 lines in `test_codegen.py` (the new
    regression test).

(c) Read the fix in situ at `x86_64.py:1005-1017`. Confirmed:

    - Classifier signatures are unchanged (`(self, ty: tir.TIRType) -> bool`).
    - The body still guards `isinstance(ty, tir.TIRScalar)` before reading
      `.name`, preserving `AttributeError`-safety on non-scalar IR types
      (TIRPtr / TIRTensor / TIRUnit / TIRArray / etc.).
    - The `in (...)` tuple check is O(2) on a fixed pair — same constant cost
      as the prior `== "i64"` comparison; no measurable perf delta.
    - The comments embed the audit reference ("Audit 28.8 cycle 19 C18-1
      (HIGH)"), the failure mode ("silently truncated to 32 bits via the else
      branch in CONST_INT/spill"), and the typecheck-canon cross-reference
      ("typecheck.py:241 ranks them at the same widening rank"). High-density
      documentation matching the `_check_float_supported` sibling pattern.

(d) Cross-verified the typecheck canon at `helixc/frontend/typecheck.py`:

    - Line 226-227: `_WIDEN_NAME_ALIASES = {..., "isize": "i64", "usize":
      "u64"}` explicitly aliases the pointer-width names to i64/u64.
    - Line 241: `_WIDEN_RANK = {..., "i64": 40, "u64": 41, "isize": 40,
      "usize": 41, ...}` ranks them at identical widening ranks.

    The cycle-19 fix is the **dual** of these typecheck canons in the backend
    classifier. The fix message's claim ("this matches typecheck.py's canon")
    is supported by direct reading of the typecheck source.

(e) Enumerated all 30 `_is_i64_type` call sites and all 3 `_is_u64_type` call
    sites by `grep -n` in `x86_64.py`. The full call-site cascade:

    | Line | Op | Pre-fix behaviour for `isize` | Post-fix |
    |-----:|----|-------------------------------|----------|
    | 986  | param spill | 32-bit spill (silent trunc) | 64-bit spill |
    | 1148 | CONST_INT | 32-bit imm (silent trunc on >2³¹) | 64-bit imm |
    | 1184-86 | LOAD/STORE wide | narrow 32-bit copy | 64-bit copy |
    | 1203-04 | CAST | cast-to/from-32 paths | cast-to/from-64 paths |
    | 1279, 1304, 1329, 1354 | ADD / SUB / MUL / DIV | 32-bit ALU | 64-bit ALU |
    | 1369, 1389, 1404, 1419, 1434, 1449, 1463, 1476 | bit/shift ops | 32-bit | 64-bit |
    | 1582 | comparison op | 32-bit cmp | 64-bit cmp |
    | 1616 | further dispatch | narrow path | wide path |
    | 1703 | indirect dispatch | narrow | wide |
    | 1716 | unwrap-result | narrow | wide |
    | 1752 | FFI_CALL arg | 32-bit param reg | 64-bit param reg |
    | 1763 | FFI_CALL return | 32-bit eax | 64-bit rax |
    | 1789, 1817, 1856, 1874 | tail dispatch | narrow | wide |

    **Every call site uses the classifier as a binary "is the value 64-bit?"
    routing predicate** — there is no site that distinguishes i64 from isize
    or relies on the strict equality. The alias extension does not over-route.

(f) Read the new regression test at `helixc/tests/test_codegen.py:477-502`:

    - Imports `FnCompiler` + `tir`, constructs `TIRScalar(name="i64")` etc.
    - Invokes `FnCompiler._is_i64_type(None, ty)` with `self=None`. The body
      reads only `ty`, never `self`, so passing `None` is safe and pins the
      classifier as a pure function on its argument. Confirmed by direct
      inspection of the body — no `self.` access.
    - Asserts both positive cases (i64 / u64 traditional names) and the alias
      cases (isize → i64, usize → u64) trip True. The assertion messages
      carry the audit stamp "C18-1".
    - Test runs in <1 ms (no full pipeline). Per `pytest`, runs in 10s
      including pytest startup. Test cost is minimal — appropriate for a
      classifier-contract regression.

(g) Ran the regression test plus the C13-1, C16-1 prior regressions + the
    array literal / array assign existing tests:
    ```
    python -m pytest helixc/tests/test_codegen.py \
        ::test_c18_1_isize_usize_recognized_as_64bit \
        ::test_c16_1_wide_array_elem_traps_at_codegen \
        ::test_array_literal_and_index \
        ::test_array_assign -v
    ```
    **4 passed in 10.01s.** No regression in the C16-1, array-literal, or
    array-assign paths.

(h) Ran a broader targeted slice:
    ```
    python -m pytest helixc/tests/test_codegen.py \
        -k "c18 or c16 or c13 or array or isize or i64 or u64 or const_int or param" -q
    ```
    **29 passed in 57.77s.** Every classifier-touching test in the codegen
    suite passes.

(i) Ran cross-component (typecheck + IR + codegen) isize/usize tests:
    ```
    python -m pytest helixc/tests/ -k "isize or usize or widening or widen" -q
    ```
    **10 passed in 22.62s.** The alias canon is consistent across typecheck,
    IR-lowering, and backend.

(j) **Adversarial probe sweep** (in-process, end-to-end through the production
    toolchain `parse → typecheck → lower → compile_module_to_elf`). The user-
    requested probe:

    ```helix
    fn main() -> i32 {
        let x: isize = 5_000_000_000_isize;
        (x >> 32) as i32
    }
    ```

    Note: the literal `5_000_000_000` without a suffix typechecks as i32 and
    fails the assignment (`type error: let 'x': declared isize but value is
    i32`); the canonical form with `_isize` suffix typechecks clean. This is
    a property of the literal-type inference (cycle-3 / C3-2 canonicalization)
    and is orthogonal to the C18-1 fix. With the suffix applied:

    - **Typecheck**: 0 hard errors.
    - **Codegen**: 4684-byte ELF produced under -O0; 4661-byte ELF under -O2
      (fold + dce). Both build successfully.
    - **Static disassembly check for the CONST_INT emit at `:1148`**: searched
      the ELF for the imm64 little-endian encoding of 5_000_000_000
      (0x12A05F200 → bytes `00 F2 05 2A 01 00 00 00`). Found at offset 4121,
      preceded by `48 B8` (REX.W + opcode B8: `mov rax, imm64`). **64-bit
      immediate confirmed.** Pre-fix the same source would have emitted
      `B8 00 F2 05 2A` (32-bit `mov eax, 0x2A05F200`), dropping the top 32
      bits of the value.

    Additional probe: **`fn f(x: isize) -> i32 { (x >> 32) as i32 }`** with a
    call site in main:

    - Static disassembly: at offset 4121 the param-spill emits `48 89 7D E0`
      = `mov [rbp-0x20], rdi` (REX.W + ModR/M for `[rbp-0x20]` <- RDI,
      64-bit store). Pre-fix would emit `89 7D E0` (no REX prefix, 32-bit
      store) — the top 32 bits of the incoming `isize` argument would be
      lost. **64-bit param spill confirmed.**

    Both reachable silent-miscompile sites named in the commit message
    (CONST_INT, fn-param spill) are empirically closed by the fix.

(k) **Blast-radius spot-checks on three downstream call sites** (selected for
    diversity):

    - ADD (`:1279`): `let x: isize = 5_000_000_000_isize; x + 1` — emits the
      i64-ALU path `mov rax, [rbp-N]; mov rcx, [rbp-M]; add rax, rcx; mov
      [rbp-K], rax` (REX.W on every instruction). Pre-fix would emit `mov
      eax`/`add eax, ecx`/`mov [...], eax`, losing the top half on every
      intermediate.
    - CAST `isize → i32` (`:1206`): correctly uses the `mov_eax_mem_rbp` /
      `mov_mem_rbp_eax` low-32-bit-truncation path. Reading the canonical
      `(x >> 32) as i32`: the SHR result is `isize`, then CAST `isize → i32`
      lands at `from_is_i64=True, to_is_i64=False`, branch at `:1206`. Same
      32-bit-low-bytes-only emit as the prior i64 → i32 path. Correct.
    - FFI_CALL arg (`:1752`): a function call with an isize argument routes
      via `INT_REGS_64[int_idx](arg_slot)`, which emits the 64-bit
      `mov rdi, [rbp-N]` (REX.W). Pre-fix would emit `mov edi, [rbp-N]`
      (32-bit, top half lost into RDI). Correct under the fix.

(l) Re-checked the commit's claim that "30+ dispatch sites... all behave
    correctly with the alias extension". By direct enumeration (e): there are
    30 `_is_i64_type` call sites and 3 `_is_u64_type` call sites. Each is
    used as a routing predicate (if-branch) selecting the 64-bit emit path.
    No site uses the classifier for type-equality comparison (e.g., `_is_i64
    _type(a.ty) and not _is_i64_type(b.ty)` for type-promotion decisions —
    those exist in the typecheck pass, not the backend). The alias extension
    is therefore **type-erasure-safe** in the backend: every consumer treats
    "i64-shape" and "isize-shape" as identical at the machine-code level,
    which matches the actual hardware contract (both are 64-bit on
    `x86_64-linux-gnu`).

(m) Cycle-counter accounting. The cycle-18 audit-B found C18-1 (HIGH) and
    reset the counter to 0/5. The cycle-19 fix-sweep closes C18-1 — but it
    does NOT itself advance the counter; the cycle-19 audits A/B/C
    collectively decide whether to advance to 1/5. This code-review audit's
    verdict is one of the three required CLEAN votes for that advance.

(n) Reviewed the diff for unrelated drift. Production files in 0803902:
    - `helixc/backend/x86_64.py` (+11 lines: comments + alias-tuple extension
      on two classifiers)
    - `helixc/tests/test_codegen.py` (+27 lines: regression test)

    **Out-of-message drift in 0803902** (noted in the user's framing as
    "agent-scaffold side effects"):
    - `_audit18_probe.py` (+145 lines, repo root)
    - `_audit18_probe2.py` (+122 lines, repo root)
    - `helixc/frontend/ast_walker.py` (+214 lines)
    - `helixc/tests/test_ast_walker.py` (+253 lines)
    - 5 docs/audit-*.md files (audit history from cycles 16-18)

    See "Below-threshold observations" §B19-1 / §B19-2 for analysis of why
    these are not findings at confidence ≥ 80.

---

## Summary table

| ID | Severity | Confidence | Component | Issue |
|----|----------|------------|-----------|-------|
| _(none)_ | — | — | — | No high-confidence issues. |

---

## Adversarial probe details

The user-requested adversarial probe and supporting probes:

### Probe 1 (user-requested) — `let x: isize = 5_000_000_000; (x >> 32) as i32`

- With unsuffixed literal: rejected at typecheck (`type error: let 'x':
  declared isize but value is i32`). The literal-type-inference does not
  promote integer literals to `isize` from the LHS annotation. This is a
  separate sema-level issue (orthogonal to C18-1) — would require
  bidirectional type inference at let-binding to flow `isize` from the LHS
  into the literal. Not in scope for cycle 19.
- With `_isize` suffix on the literal: typecheck-clean. Codegen produces a
  4684-byte ELF under -O0 and a 4661-byte ELF under -O2.
- **Static disassembly verifies the 64-bit emit path**: the imm64 encoding
  of 5_000_000_000 (LE bytes `00 F2 05 2A 01 00 00 00`) is present at
  offset 4121 of the ELF, preceded by the `48 B8` REX.W + B8 prefix
  identifying `mov rax, imm64`. The top 32 bits (0x00000001) are
  unambiguously preserved. Pre-fix the encoding would have been the
  truncated `B8 00 F2 05 2A` 32-bit form, losing the high bit.

### Probe 2 — fn-param spill for isize

`fn shift_top(x: isize) -> i32 { (x >> 32) as i32 }` called from `main()`:

- Static disassembly: `48 89 7D E0` at offset 4121 of the `shift_top`
  prologue: REX.W + ModR/M for `mov [rbp-0x20], rdi`. 64-bit param spill
  confirmed. Pre-fix this would have been `89 7D E0` (no REX, 32-bit
  store), losing the top 32 bits of RDI.

### Probe 3 — full test-suite slice covering classifier-affected paths

`pytest -k "c18 or c16 or c13 or array or isize or i64 or u64 or const_int
or param" -q`: **29 passed in 57.77s.** No regression.

### Probe 4 — cross-component isize/usize/widening tests

`pytest -k "isize or usize or widening or widen" -q`: **10 passed in
22.62s.** Alias canon consistent across typecheck/IR/backend.

### Probe 5 — classifier-contract direct test (the new regression test)

The new `test_c18_1_isize_usize_recognized_as_64bit` runs in 10s and
asserts:
- `_is_i64_type(None, TIRScalar("i64"))` is True
- `_is_i64_type(None, TIRScalar("isize"))` is True
- `_is_u64_type(None, TIRScalar("u64"))` is True
- `_is_u64_type(None, TIRScalar("usize"))` is True

The use of `self=None` is sound: the classifier bodies do not read `self`.
Verified by direct inspection of the function body.

---

## Verification of the cycle-18 fix under the new probes

The cycle-19 fix is the 2-line classifier extension at `x86_64.py:1011, 1017`,
plus a 27-line regression test. The probes validate:

1. **Both commit-message-named miscompile sites are closed empirically**:
   Probe 1 (CONST_INT 5e9 isize) emits `mov rax, imm64` with the full
   8-byte value; Probe 2 (fn-param isize spill) emits `mov [rbp-N], rdi`
   with REX.W. Pre-fix encoding bytes are NOT present in the post-fix ELF.

2. **30 `_is_i64_type` consumers + 3 `_is_u64_type` consumers all route
   isize/usize to the 64-bit path**: per the call-site enumeration in
   §(e). No consumer uses the classifier for type-equality or
   promotion-direction decisions, so no over-routing exists. Backend
   type-erasure of `isize` → `i64-shape` is semantically identical to
   the hardware contract on x86_64.

3. **No regression in i64/u64 native paths**: the 29-test targeted slice
   includes `test_i64_const`, `test_i64_arith`, `test_u64_const`, etc.
   (per `pytest -k` matching), all passing.

4. **Alias canon consistent with typecheck**: typecheck.py:226-227 aliases
   isize → i64, usize → u64 in `_WIDEN_NAME_ALIASES`; line 241 ranks them
   identically in `_WIDEN_RANK`. The backend fix is the dual of these
   canons. No frontend/backend divergence.

5. **Regression test is a sound classifier-contract pin**: invokes the
   classifier as a pure function on the type argument; trips on every
   alias case; carries audit-stamp marker. Adds <1ms to test time.

6. **No drift in the production code**: diff in 0803902 is exactly the
   2-line alias extension + comment block + the regression test. The
   other files in 0803902 are non-production (see §B19-1, §B19-2 below).

---

## Why no findings at ≥ 80

1. **Fix is minimal, correct, and aligned with the typecheck canon**: the
   alias extension is the smallest possible change that closes C18-1.
   `("i64", "isize")` and `("u64", "usize")` map 1:1 to typecheck.py's
   `_WIDEN_NAME_ALIASES` entries. No over-extension; no under-extension.

2. **Call-site cascade verified by enumeration**: 30 `_is_i64_type` + 3
   `_is_u64_type` consumers, each a binary routing predicate selecting
   between 32-bit and 64-bit emit paths. The alias extension routes
   isize/usize to the 64-bit path universally — which is the correct
   x86_64 ABI behaviour for pointer-width types. No call site requires
   distinguishing i64 from isize.

3. **Static disassembly confirms the silent-miscompile is closed**: both
   the CONST_INT 5_000_000_000 case and the fn-param spill case emit the
   REX.W-prefixed 64-bit instruction sequences in the post-fix ELF.
   Pre-fix byte patterns are absent.

4. **Targeted test slice + cross-component tests all green**: 29 + 10 =
   39 relevant tests pass with zero regressions. The new C18-1 test pins
   the classifier contract uniquely.

5. **No state mutation, no exception-safety concern, no allocation**: the
   classifier is a pure function; the alias extension does not change
   its return-type or raise behaviour. Defensive `isinstance(ty,
   tir.TIRScalar)` guard preserved.

6. **Documentation density matches sibling pattern**: the inline comments
   embed the audit reference, failure mode, and typecheck cross-reference,
   matching `_check_float_supported`'s convention.

---

## Below-threshold observations from this cycle

### B19-1 — Commit 0803902 includes 267 LOC of `_audit18_probe*.py` repo-root scripts unrelated to the C18-1 fix (conf 55, NEW)

**Location**: `_audit18_probe.py` (+145), `_audit18_probe2.py` (+122),
both at repo root in commit 0803902.

**Observation**: the commit message names only the classifier extension
+ regression test as the deliverable. The two probe scripts are ad-hoc
audit-investigation scripts (Python snippets used by the cycle-18 audit
agent to inspect the backend classifiers). They are not part of the
build, not part of the test suite, not imported by anything, and live
at repo root rather than under `helixc/` or `scripts/`.

**Why this is not a finding at ≥ 80**: the user explicitly flagged this
in the audit brief ("not part of the C18-1 fix... if their presence
raises code-quality concerns at confidence ≥ 80, flag. Otherwise note
below threshold"). The follow-on commit e53e510 ("chore: remove stale
audit-cycle-18 probe scripts") deletes them with a clean rationale.
They are commit-hygiene drift in 0803902 alone but are recoverable by
the immediate next commit and do not affect the C18-1 fix's correctness.
Confidence 55 (a code-reviewer would note "this looks like agent state
leaking into the commit" but would not block).

**Forward note**: future fix-sweep commits should either gitignore
`_audit*_probe*.py` patterns at repo root, or the audit agent should
deposit its scratch scripts to a `runtime/`-style ignored path.

---

### B19-2 — Commit 0803902 includes 467 LOC of `ast_walker.py` + `test_ast_walker.py` with no production consumers at this commit (conf 50, NEW)

**Location**: `helixc/frontend/ast_walker.py` (+214), `helixc/tests/
test_ast_walker.py` (+253), both new in 0803902.

**Observation**: `ast_walker.py` is a legitimate Stage 28.8.2 deliverable
(shared AST traversal infrastructure — `ASTVisitor` base class using
`dataclasses.fields(node)` introspection, replacing hand-rolled per-pass
walkers in `panic_pass` / `unsafe_pass` / `deprecated_pass` / `grad_pass`
/ `struct_mono`). The accompanying `test_ast_walker.py` has 10 tests, all
passing. The module is well-documented (53-line docstring explaining the
design choices) and orthogonal to the C18-1 classifier fix.

**However**, at commit 0803902 in isolation, the 5 named consumers do
NOT yet import `ast_walker.ASTVisitor` — those import-side refactors land
in subsequent commits 7649c15, a04036b, 9436810, 44f17e6. At 0803902,
`ast_walker.py` is an orphan library with a test (no production
consumers). The cycle-19 fix-sweep commit message does not mention the
Stage 28.8.2 work.

**Why this is not a finding at ≥ 80**:
- The code itself is correct, tested, and well-documented (10/10 tests
  pass; introspective dataclass-based walker is sound).
- The orphan status at 0803902 is temporary; the wire-up commits land
  within 4-5 commits.
- The commit-hygiene concern is real but cosmetic: a fix-sweep commit
  bundling unrelated infrastructure is a process/review-process miss,
  not a code-correctness defect. A code-reviewer would request a
  separate commit for the Stage 28.8.2 infrastructure but would not
  block the C18-1 closure.

Confidence 50 (commit scope drift, no functional impact).

**Forward note**: the cycle-19 audit silent-failures lens may want to
verify whether the `ast_walker` introspective walker has full coverage
of every AST `dataclass` schema's field set, especially for `TyNode`-
typed fields (which the docstring says are intentionally NOT walked by
default — that's a design choice, but worth pinning as a regression
test if not already).

---

### B19-3 — Adversarial probe surfaced a literal-typing gap (unsuffixed integer literal does not promote to isize from LHS annotation) (conf 45, NEW)

**Location**: literal-type-inference in `helixc/frontend/typecheck.py`
(line not pinpointed in this audit).

**Observation**: the user-requested adversarial probe
```
fn main() -> i32 {
    let x: isize = 5_000_000_000;
    (x >> 32) as i32
}
```
fails typecheck with `type error: let 'x': declared isize but value is
i32`. With the `_isize` suffix on the literal (`5_000_000_000_isize`),
typecheck is clean. The typechecker does not perform bidirectional
inference at let-binding to flow the declared type into the unsuffixed
literal.

This means a user writing pure Rust-style code (`let x: isize = 5e9;`)
gets a typecheck rejection rather than a successful compile-then-trunc
miscompile. From a safety standpoint this is the **safer** failure mode
(typecheck rejection > silent miscompile). But from a UX standpoint, the
diagnostic is confusing for a literal that fits in isize.

**Why this is not a finding at ≥ 80**: the typecheck rejection is safe
(not a miscompile); it's a feature-gap / UX issue, not a defect.
Bidirectional inference at let-binding is a Stage-29+-class feature.
Cycle-19's scope is closing C18-1, and this issue is independent.

**Forward note**: a future audit could flag the literal-inference gap
as a UX issue under the silent-failures lens (would land as a MEDIUM
"surprising-but-safe" finding, not HIGH).

---

### B19-4 — Regression test invokes classifier as unbound method with `self=None` (conf 30, NEW)

**Location**: `helixc/tests/test_codegen.py:494-501`.

```python
assert FnCompiler._is_i64_type(None, i64) is True
```

**Observation**: the test invokes the instance method as `FnCompiler._is_i64
_type(None, ...)`, passing `None` for `self`. This works because the
classifier body never reads `self.*` — but the contract that it does not
read `self` is not pinned anywhere except by direct code inspection. A
future change that adds `self.target` (e.g., for 32-bit-target support)
would silently break this test pattern, and the test would `AttributeError`
on `None.target` rather than fail with a clean diagnostic.

**Why this is not a finding at ≥ 80**: cosmetic. The test correctly pins
the classifier contract today, and a future maintainer modifying the
classifier to read `self` would see the test fail (loud failure, not
silent miscompile). Alternative: instantiate `FnCompiler` with a minimal
fn / module pair, or refactor the classifier as a `@staticmethod` /
module-level function. Either would tighten the contract. Confidence 30.

---

### Carryover from prior cycles (unchanged)

The following carryover items from cycles 10-17 remain unchanged and are
NOT re-flagged per user directive:

- B10-x family (empty-string / nested-prefix / whitespace edge cases for
  `_emit_env_error`, raise-message convention) — all conf < 50.
- B14-2 / B15-1 (dce.py docstring partial-enumeration of
  `SIDE_EFFECT_KINDS`, conf 30).
- B17-1 (nested array literal `[[T; N]; M]` does not nest at IR — would
  decay to `i32` placeholders, conf 60). cf9cf7e ("Audit 28.8 cycle 19+:
  close cycle-18 audit-C C18-1 (nested-aggregate silent decay)") may
  have closed this at HEAD; this audit is at 0803902 so the carryover
  remains as recorded in cycle 17.
- B17-2 (LOAD_ELEM index-operand type non-coverage, conf 35).
- Cycle-16 forward notes (Value.ty not frozen, Op.results: list
  invariant, _alloc_array elem_size unused parameter, PTX _format_param
  hard-coded .b64) — all Stage-29-class.

---

## Cycle 19 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff zero
findings of ANY severity at confidence ≥ 80.**

This audit (Audit C, code-review) finds **0 findings at confidence ≥ 80**.

The other two cycle-19 audits (silent-failures, type-design) will each
render their own verdict. If all three render CLEAN, cycle 19 advances
the counter to 1/5.

**Counter status (5-clean-consecutive gate)**:
- Was 0/5 after cycle 18 (reset by C18-1 HIGH finding).
- Cycle 19 code-review (this audit): **CLEAN**. Contributes one of the
  three CLEAN votes required for cycle 19 to advance the counter.
- If silent-failures and type-design also CLEAN: counter 0/5 → 1/5.
- Four more clean cycles after cycle 19 (cycles 20, 21, 22, 23) then
  complete the gate.

The severity trend across cycles, against the strict-criterion bar:
- Cycle 1-6: not clean
- Cycle 7-12: clean (counter advanced to 3/5)
- Cycle 13: 1 HIGH (C13-1) — not clean → reset to 0/5
- Cycle 14: clean → 1/5
- Cycle 15: clean → 2/5
- Cycle 16: 1 HIGH (C16-1) — not clean → reset to 0/5
- Cycle 17: clean → 1/5
- Cycle 18: 1 HIGH (C18-1) — not clean → reset to 0/5
- Cycle 19 code-review (this audit): 0 — CLEAN-vote contributed

---

## Verdict

**CLEAN** under Audit C (code-review) on the cycle-19 fix-sweep at 0803902.
The `_is_i64_type` / `_is_u64_type` alias extensions are minimal, correct,
and aligned with the typecheck.py `_WIDEN_NAME_ALIASES` canon. The 30 +
3 call-site cascade routes isize/usize to the 64-bit emit path universally,
matching the x86_64 ABI contract. Static disassembly verifies that both
silent-miscompile sites named in the commit message (CONST_INT 5e9 isize,
fn-param isize spill) emit REX.W-prefixed 64-bit instructions in the
post-fix ELF; pre-fix byte patterns are absent. Targeted regression
testing (39 tests across codegen + cross-component) passes with zero
regressions. No high-confidence code-review concern at this commit.

Forwarded for future-cycle attention:
- B19-1 (`_audit18_probe*.py` repo-root scripts in fix-sweep commit,
  conf 55) — already closed by follow-on commit e53e510.
- B19-2 (`ast_walker.py` orphan-library introduction in fix-sweep
  commit, conf 50) — already wired up by follow-on commits 7649c15+.
- B19-3 (literal-type inference gap: unsuffixed `5_000_000_000` does not
  promote to isize from LHS annotation, conf 45) — UX issue, not defect.
- B19-4 (regression test invokes classifier with `self=None`, conf 30) —
  cosmetic.
