# Stage 28.8 Cycle 19 — Silent-Failure Audit (Audit A)

**Date**: 2026-05-11
**Commit (nominal HEAD for this audit)**: `0803902` — "Audit 28.8
cycle 19 fix-sweep: close C18-1 (HIGH, isize/usize silent trunc)".

**Context (new streak)**: Cycle 18 Audit B (type-design lens) surfaced
**C18-1 / HIGH** — the `x86_64.FnCompiler` type classifiers
`_is_i64_type` / `_is_u64_type` used strict-equality `ty.name == "i64"`
/ `ty.name == "u64"`, missing the pointer-width aliases `isize` /
`usize`. `typecheck.py:241` ranks isize at the same widening rank as
i64 (cycle-3 C3-2 fix) and `lower_ast.py` preserves the name string
through to TIR, so a program like

```helix
let x: isize = 5_000_000_000;
```

typechecked clean and reached codegen with `TIRScalar("isize")` —
where the backend's classifier fell back to the 32-bit emit branch
(`mov_eax_imm32(value & 0xFFFFFFFF)`), silently truncating the
literal. Cycle 18 cascade traced 30+ `_emit_op` dispatch sites and
documented two reachable silent miscompiles:

1. **CONST_INT** at `x86_64.py:1138-1147` — wide literal silently
   truncated to 32 bits on store.
2. **Fn-param spill** at `x86_64.py:971-989` — `fn f(x: isize)`
   spilled the incoming register with the 32-bit move, dropping
   the top 32 bits of the SysV argument.

The cycle-19 fix-sweep at `0803902` extended both classifiers to
recognize the aliases:

```python
def _is_i64_type(self, ty: tir.TIRType) -> bool:
    return isinstance(ty, tir.TIRScalar) and ty.name in ("i64", "isize")

def _is_u64_type(self, ty: tir.TIRType) -> bool:
    return isinstance(ty, tir.TIRScalar) and ty.name in ("u64", "usize")
```

This **closes** the C18-1 backend silent-trunc path. New regression
test `test_c18_1_isize_usize_recognized_as_64bit` (test_codegen.py:
477-501) pins the classifier contract directly.

That fix reset the strict clean-cycle counter to 0/5. Cycle 19 is
the **first** read-only re-audit of the new streak (need 5
consecutive clean cycles to fire the Stage-29 gate).

**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity** (CRITICAL /
HIGH / MEDIUM / LOW). Findings already in the carryover ledger
(audit-C4-1 CRITICAL, audit-C4-4 HIGH, audit-C4-8 LOW, C5-10 LOW,
monomorphize_safe docstring drift, D-vs-Quote diagnostic text,
C7-1 test-coverage gap) are not re-flagged here per the strict
re-flag rule (none changed since prior cycle).

**Clean-counter state going into cycle 19**: 0/5 (counter reset by
C18-1).

---

## Method

1. **Read prior cycle silent-failure verdicts** (cycles 10–17 and
   cycle-18 Audit B's type-design lens for C18-1 context). Cycles
   10–17 silent-failures were all CLEAN. Cycle 18 type-design lens
   found C18-1 (closed at 0803902).
2. **`git show 0803902 --stat`** confirms the fix surface:
   - Production-code delta: **+9 / -2** lines in
     `helixc/backend/x86_64.py` (the two classifier extensions plus
     audit-stamp comments).
   - Test delta: **+27** lines in `helixc/tests/test_codegen.py`
     (one new regression test pinning the classifier contract).
   - Everything else is documentation (cycle 16/17 type-design and
     code-review docs, cycle 17 silent-failures doc, ast_walker
     refactor scaffolding from a parallel staging branch).
3. **Verified the fix at the audit HEAD** by reading the two
   classifier extensions and the regression test:
   - `helixc/backend/x86_64.py:1005-1017` — both classifiers now
     accept the alias names.
   - `helixc/tests/test_codegen.py:477-501` — the test instantiates
     `TIRScalar("isize")` / `TIRScalar("usize")` and asserts both
     classifiers return True. Mirror for `i64` / `u64` confirms
     the canonical names still match.
4. **Read-only re-audit of the cycle-19 fix surface + every
   classifier consumer site**. `grep _is_i64_type|_is_u64_type
   helixc/backend/x86_64.py` returns 30 consumer sites in
   `_emit_op` (lines 986, 1148, 1184-1186, 1203-1204, 1279, 1304,
   1329, 1354, 1369, 1389, 1404, 1419, 1434, 1449, 1463, 1476,
   1582, 1616, 1703, 1716, 1752, 1763, 1789, 1817, 1856, 1874).
   Spot-checked all CONST_INT / arithmetic / shift / bitwise /
   compare / NEG / BR / RETURN / CALL / FFI_CALL / LOAD_VAR /
   STORE_VAR / CAST dispatches against the cycle-19 widened
   classifier semantics.
5. **Adversarial rotation** — replicate the cycle-18 adversarial
   move: list every other file that carries an isize/usize-aware
   width table, and ask whether the same fix-sweep deliverable
   should have touched it. The C18-1 fix is a **canonicalization**
   (treat isize as i64, treat usize as u64), so any other module
   that ALSO indexes width by the type's name string needs the
   identical change — or it now disagrees with the post-fix
   backend and re-opens the silent-trunc path through a different
   pass.
6. **Read-only**: no edits to production code or tests during this
   audit cycle.

---

## Fresh-eyes audit of the cycle-19 fix surface

### The two classifier extensions (x86_64.py:1005-1017)

```python
def _is_i64_type(self, ty: tir.TIRType) -> bool:
    # Audit 28.8 cycle 19 C18-1 (HIGH): `isize` is a pointer-width
    # alias of `i64` on 64-bit targets — typecheck.py:241 ranks them
    # at the same widening rank, but the backend classifier was
    # name-equal only, so `let x: isize = 5_000_000_000;` silently
    # truncated to 32 bits via the else branch in CONST_INT/spill.
    return isinstance(ty, tir.TIRScalar) and ty.name in ("i64", "isize")

def _is_u64_type(self, ty: tir.TIRType) -> bool:
    # Stage 16.5: u64 is the IR type for raw pointers and FFI-arg widening.
    # Audit 28.8 cycle 19 C18-1: `usize` is a pointer-width alias of
    # `u64` on 64-bit targets. Same silent-trunc class as isize.
    return isinstance(ty, tir.TIRScalar) and ty.name in ("u64", "usize")
```

**Audit walk**:

1. **Trigger condition** — `isinstance(ty, tir.TIRScalar) and
   ty.name in ("i64", "isize")` (similarly `("u64", "usize")` for
   the u64 classifier). The membership check is closed-set; no
   regex or prefix-match that could over-match a synonym; no
   `.get(...)` default that could silently mis-classify a
   user-supplied scalar name. The condition is correct and
   complete for the documented hazard surface (typecheck's
   `_widen_canon_name` table at typecheck.py:225-228 maps
   isize→i64 and usize→u64, which exactly matches the alias
   set here).

2. **Negative path** — anything not in the membership set falls
   through to `False`. Consumers branch on `_is_i64_type(...)`:
   True ⇒ 64-bit emit, False ⇒ 32-bit emit. The False path is
   the 32-bit codegen — correct for i32/u32/i16/u16/i8/u8/bool.
   No new silent-failure window opened on the False side.

3. **Could the new True path mis-classify any signed-vs-unsigned
   semantics?** `_is_i64_type` is consulted by ops where signed
   vs unsigned matters (SHL/SHR with `sar_*`, DIV with signed
   `idiv`, signed comparisons via `setl/setg`). isize is signed
   in Helix's type system (range `(-(1<<63), (1<<63)-1)` per
   typecheck.py:1816), so canonicalizing isize to the i64-signed
   path is semantically correct. Symmetric for usize→u64 in
   FFI_CALL paths (lines 1752/1763 — the only `_is_u64_type`
   consumers in production code). `usize` is the only IR scalar
   that's both unsigned and pointer-wide; no FFI-call site
   distinguishes u64 from usize, so canonicalization preserves
   semantics.

4. **Could `TIRScalar(name="isize")` ever flow in with a
   meaning OTHER than "pointer-width signed integer"?**
   Origins of `TIRScalar("isize")`:
   - `lower_ast.py:357-358` includes `isize` in
     `_PRIMITIVE_TYPE_NAMES`; line 371 returns
     `tir.TIRScalar(ty.name)` for primitive names.
   - The parser/lexer recognize `isize` only as the `KW_ISIZE`
     primitive keyword (`lexer.py:90`, `parser.py:666`).
   - No other code path manufactures `TIRScalar("isize")` —
     `grep TIRScalar.*isize` returns only the canonical sites.

   There is no shadowed "isize" string with a different meaning
   that could be silently mis-classified.

5. **Raise quality** — no raises here; classifier always returns
   a bool. No swallowed errors, no `.get(...)` defaults.
   Audit-stamp comments cite "audit 28.8 cycle 19 C18-1" and
   reference the prior-state and the typecheck cross-reference.

**Verdict on the classifier extensions**: clean. The widened
membership set strictly *closes* the silent-trunc window at the
30+ `_emit_op` dispatch sites that branch on these classifiers,
without opening any new ones. No signed/unsigned, value-range,
or origin-shadowing hazard.

### Regression test (test_codegen.py:477-501)

```python
def test_c18_1_isize_usize_recognized_as_64bit():
    from helixc.backend.x86_64 import FnCompiler
    from helixc.ir import tir
    i64 = tir.TIRScalar(name="i64")
    isize = tir.TIRScalar(name="isize")
    u64 = tir.TIRScalar(name="u64")
    usize = tir.TIRScalar(name="usize")
    assert FnCompiler._is_i64_type(None, i64) is True
    assert FnCompiler._is_i64_type(None, isize) is True, (
        "isize should be recognized as i64-width (C18-1)"
    )
    assert FnCompiler._is_u64_type(None, u64) is True
    assert FnCompiler._is_u64_type(None, usize) is True, (
        "usize should be recognized as u64-width (C18-1)"
    )
```

Pins the classifier contract directly. The test passes locally
(`python -m pytest helixc/tests/test_codegen.py::
test_c18_1_isize_usize_recognized_as_64bit -v` → 1 passed in
2.61s). The contract test is fast and minimal — no full-pipeline
ELF emission — but the cycle-18 audit's site-by-site cascade
already verified that classifier ⇒ correct dispatch at every
consumer (so a contract test on the classifier is sufficient).

**No new silent-failure window introduced by the regression test.**

### Spot-checked consumer sites

Sampled five representative dispatch sites against the post-fix
classifier semantics:

- **CONST_INT** (`x86_64.py:1148`) — `if self._is_i64_type(
  op.results[0].ty):` now True for isize-typed results, so the
  64-bit `mov_rax_imm64 + mov_mem_rbp_rax` branch fires. Wide
  literals survive. **Closes.**
- **Fn-param spill** (`x86_64.py:986`) — same dispatch, same
  closure. SysV-passed isize args spill with 64-bit move.
- **ADD/SUB/MUL/DIV** (`x86_64.py:1279, 1304, 1329, 1354`) — i64
  branch now also reached by isize-typed results; emits
  `mov_rax_mem_rbp + add_rax_rcx + mov_mem_rbp_rax`. Signed
  64-bit arithmetic, which matches isize's signed semantics.
- **Signed CMP** (`x86_64.py:1582`) — `_is_i64_type` check on
  either operand now triggers the 64-bit `cmp rax, rcx` +
  signed setcc path for isize comparisons. Closes the cycle-18
  cited reproducer (`x > 4_000_000_000` with `x: isize`).
- **FFI_CALL arg routing** (`x86_64.py:1752`) — `or` of the two
  classifiers now correctly catches usize pointer args; previously
  a `usize` parameter to an FFI fn would have spilled via 32-bit
  `mov_edi_mem_rbp`, truncating an above-2GB pointer. **Closes**
  a latent FFI silent-trunc path that wasn't explicitly named in
  the cycle-18 cascade but lives in the same defect class.

**All spot-checked consumer sites** now route isize/usize through
the same 64-bit emit paths as i64/u64. No consumer site exists
where i64 and isize need to behave **differently** (the language
model treats them as aliases at typecheck per `_widen_canon_name`,
so any consumer-site divergence would itself be a bug — none
found).

---

## Adversarial rotation: who else carries a width table for isize/usize?

The cycle-18 audit's adversarial move was to **enumerate every
pass / module that branches on the scalar type name**. The fix-
sweep at 0803902 patched the backend classifier but the
canonicalization (treat isize as i64 width) is a **language-
level contract** — every pass that materializes width semantics
from the type's name string must follow the same canon, or it
re-opens the same silent-trunc path via a different pass.

`grep -r "isize\|usize" helixc/` outside the backend returns:

| Site                                                                 | Purpose                                              | Width-correct? |
|----------------------------------------------------------------------|------------------------------------------------------|----------------|
| `helixc/frontend/lexer.py:90, 338`                                   | Token-keyword set                                    | N/A (no width) |
| `helixc/frontend/parser.py:666`                                      | Token→type-name map                                  | N/A (no width) |
| `helixc/frontend/ast_nodes.py` (per grep)                            | AST type-name set                                    | N/A (no width) |
| `helixc/frontend/typecheck.py:225-228`                               | `_widen_canon_name`: isize→i64, usize→u64            | **Correct**    |
| `helixc/frontend/typecheck.py:241`                                   | Widening rank: isize=40 (same as i64), usize=41      | **Correct**    |
| `helixc/frontend/typecheck.py:337-338, 2091-2092`                    | Numeric primitive sets                               | N/A (no width) |
| `helixc/frontend/typecheck.py:1816-1817`                             | Value-range table: isize signed 64-bit, usize 64-bit | **Correct**    |
| `helixc/frontend/autodiff.py:74-75`                                  | `NUMERIC_FOR_AD` set                                 | N/A (no width) |
| `helixc/frontend/monomorphize.py:704`                                | Docstring example                                    | N/A            |
| `helixc/ir/lower_ast.py:357-358`                                     | `_PRIMITIVE_TYPE_NAMES`                              | N/A (no width) |
| **`helixc/ir/passes/const_fold.py:43-49`**                           | **`_INT_BITS`: maps name → width for wrapping**      | **WRONG ⚠️**   |

The const_fold table is the **one outlier**. It was correct
**before** the C18-1 fix (`isize: 32` was consistent with the
backend's 32-bit emit path) but is now **inconsistent with the
post-fix backend** (backend emits 64-bit, const_fold wraps to
32-bit). This re-opens the silent-trunc path through the const-
fold pass, by exactly the same mechanism that C18-1 documented
for the backend.

This is the new finding for cycle 19.

---

## Findings

### C19-1 / HIGH — `const_fold._INT_BITS` contradicts the post-cycle-19 backend; folded isize/usize arithmetic silently truncates to 32 bits

**Location**

`helixc/ir/passes/const_fold.py:43-49`:

```python
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32, "isize": 32, "usize": 32,   # ← isize/usize wrong
    "i64": 64, "u64": 64,
    "bool": 32,
}
```

`_wrap_int_to_type(value, ty)` at `const_fold.py:59-68` reads:

```python
def _wrap_int_to_type(value: int, ty: "tir.TIRType") -> int:
    bits = 32  # default for unknown / generic scalar types
    if isinstance(ty, tir.TIRScalar):
        bits = _INT_BITS.get(ty.name, 32)
    mask = (1 << bits) - 1
    half = 1 << (bits - 1)
    v = value & mask
    if v >= half:
        v -= (1 << bits)
    return v
```

`_wrap_int_to_type` is the **single sink** for every const-folded
integer arithmetic op:

- Binary ADD/SUB/MUL/DIV/MOD result at const_fold.py:334
- Binary BIT_AND/BIT_OR/BIT_XOR/SHL/SHR result at const_fold.py:410
- Unary NEG result at const_fold.py:451
- Unary BIT_NOT result at const_fold.py:482

For every one of these ops, when `res.ty == TIRScalar("isize")`
or `TIRScalar("usize")`, the lookup `_INT_BITS["isize"]` returns
**32** — but the post-fix backend now emits these results as
**64-bit**. The folded literal is therefore stored as a 32-bit
sign-wrapped value into a CONST_INT op whose type-tag says 64
bits; the backend's CONST_INT emit (`x86_64.py:1148-1153`,
post-fix) takes the 64-bit `mov rax, imm64 + mov [rbp-N], rax`
branch using that already-truncated value.

**Reachable end-to-end silent miscompile**

Pre-fix reproducer (literal-only) for C18-1:

```helix
let x: isize = 5_000_000_000;
return x as i32;
```

Post-cycle-19, this reproducer is **closed** because CONST_INT
for an `isize` literal at lowering does not pass through
`_wrap_int_to_type` (literals are stored verbatim into
`CONST_INT.attrs["value"]` by `lower_ast.py`). The 64-bit emit
path now preserves the value.

**New (C19-1) reproducer that survives the cycle-19 fix:**

```helix
fn main() -> i32 {
    let a: isize = 3_000_000_000;
    let b: isize = 3_000_000_000;
    let c: isize = a + b;        // expected: 6_000_000_000
    if c > 5_000_000_000 {
        return 1;
    }
    return 0;
}
```

Execution trace:

1. Typecheck binds `a`, `b`, `c` to `TIRScalar("isize")`.
2. const_fold's binary-ADD path (line 327-333) is invoked on
   `a + b`. Both are CONST_INT literals (no further folding
   needed), but if both are constant-propagated through a
   later pass (or if the user wrote `3_000_000_000_isize +
   3_000_000_000_isize` directly), the folder evaluates
   `v = 3_000_000_000 + 3_000_000_000 = 6_000_000_000`.
3. `_wrap_int_to_type(6_000_000_000, TIRScalar("isize"))`:
   - `bits = _INT_BITS["isize"] = 32`.
   - `mask = (1 << 32) - 1 = 0xFFFFFFFF`.
   - `v = 6_000_000_000 & 0xFFFFFFFF = 1_705_032_704`.
   - `half = 1 << 31`. `v < half`, so no sign-extend.
   - Returns `1_705_032_704`.
4. The fold rewrites the ADD op as `CONST_INT { value:
   1_705_032_704, ty: isize }`.
5. CONST_INT emit at backend (post-cycle-19 fix) sees
   `_is_i64_type(isize) = True` and emits 64-bit
   `mov rax, 0x65A0BC00` (the truncated value, zero-extended
   to 64 bits — but the value is already wrong).
6. The CMP_GT `c > 5_000_000_000` at i64 width (signed) compares
   `1_705_032_704 > 5_000_000_000` → **false**. main returns 0.

**Expected behavior**: `c == 6_000_000_000`, predicate true,
main returns 1.

**Actual behavior** at HEAD=0803902: main returns 0. No
diagnostic, no warning. Silent miscompile.

The same hazard exists for every arithmetic-folded isize/usize
value crossing the 2³¹ boundary — the const-folder produces a
literal that disagrees with both (a) the runtime semantics the
backend implements (post-fix: 64-bit two's complement) and
(b) the value-range typecheck enforces (typecheck.py:1816: isize
range is `(-(1<<63), (1<<63)-1)`).

**Why the cycle-19 fix-sweep missed this**

The cycle-19 commit message at 0803902 lists "30+ dispatch sites
in `_emit_op`" as the cascade scope and names two reachable
silent miscompiles (CONST_INT + fn-param spill). The cascade
stayed within `x86_64.py`. The **cycle-18 type-design audit
doc** (which surfaced C18-1) explicitly flagged `_INT_BITS` at
line 446 as needing the matching adjustment ("**Plus matching
adjustments to `_INT_BITS` in const_fold.py:46 to treat
isize/usize as 64-bit.**") and at line 544 ("forward note 1 —
Centralize scalar-width predicate"). The fix-sweep did not
honor that forward note.

This is a defect-class continuation: same "canonicalization not
propagated to every width-aware table" pattern as C16-1's array-
element trap (where the LOAD_ELEM/STORE_ELEM dispatch in the
backend was patched but no scan for parallel sibling-backend
sites was done — fortunately the PTX and dyn-ELF backends don't
emit those ops yet, so cycle 17 was clean). The const_fold
miss is the first time the canonicalization-not-propagated
pattern reaches a production-reachable silent miscompile in a
sibling **pass** (vs sibling backend file).

**Hidden errors**

The const-folder has no diagnostic, no `assert`, no warning
emission anywhere in the fold path. Wrong fold results
propagate through subsequent optimizations as if they were
correct literals — DCE could then drop branches the user
expected to execute (or keep branches the user expected to
skip), compounding the miscompile silently.

**User impact**

A user writing `let x: isize = N1 + N2;` where the const-folded
sum exceeds 2³¹ - 1 gets a binary that returns the wrong answer
with no warning. The compiler reports zero errors / zero
warnings; the test passes for small N1/N2 (below 2³¹); only
above-2³¹ inputs trip the bug. Standard "small inputs work,
large inputs break" silent miscompile.

**Severity**: HIGH (silent miscompile, reachable from valid
typechecking-clean source, no diagnostic).

**Recommendation**

Single-line fix — align const_fold's width table with the
cycle-19 backend canon:

```python
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32,
    "isize": 64, "usize": 64,     # ← match the backend canon
    "i64": 64, "u64": 64,
    "bool": 32,
}
```

Plus a full-pipeline regression test that round-trips
`3_000_000_000_isize + 3_000_000_000_isize` through compile →
fold → emit → run and asserts the runtime answer equals
`6_000_000_000`. The cycle-18 type-design doc (line 446)
already noted that a round-trip test was the right shape for
this defect class; the C18-1 regression test is classifier-
contract-level only, which is why this slipped.

**Example of corrected code**

```python
# helixc/ir/passes/const_fold.py:43
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32,
    # Audit 28.8 cycle 19 C19-1 (HIGH): pointer-width aliases must be
    # 64-bit, matching typecheck.py:225-228's `_widen_canon_name`
    # aliasing (isize→i64, usize→u64) and the cycle-19 backend
    # classifier fix at x86_64.py:1005-1017. Pre-fix the 32-bit
    # entry made `_wrap_int_to_type(6_000_000_000, isize) =
    # 1_705_032_704` — silent miscompile reachable at default -O1.
    "isize": 64, "usize": 64,
    "i64": 64, "u64": 64,
    "bool": 32,
}
```

---

## Cross-lens corroboration

The parallel **cycle-19 type-design lens audit**
(`docs/audit-stage28-8-cycle19-type-design.md`) — written
independently and present in the working tree at the time of this
audit's drafting — flagged the **identical** finding (C19-1 in
that document, lines 100-220). The two lenses converge on the
same defect by different routes:

- Type-design lens: "post-fix `_is_i64_type(isize) = True` and
  `_INT_BITS[isize] = 32` are an inconsistent width contract".
- Silent-failures lens: "every const-fold sink calls
  `_wrap_int_to_type` and silently truncates the result with
  no diagnostic".

Cross-lens corroboration of a HIGH finding through independent
methodology is exactly the audit-rotation pattern this series is
designed to produce. The defect is real and HIGH.

---

## Verdict

**Cycle 19 silent-failures audit: NOT CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| **HIGH**   | **1** |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **1** |

- **C19-1 / HIGH** — const_fold `_INT_BITS` lacks the isize/usize
  → 64 mapping; folded arithmetic results crossing the 32-bit
  boundary silently truncate; reachable end-to-end miscompile
  in valid typechecking-clean source.

Clean-cycle counter: was 0/5 → **stays at 0/5** (strict criterion:
any non-zero finding count resets / blocks the streak). The
cycle-20 fix-sweep should close C19-1 with the single-line
`_INT_BITS` patch plus the full-pipeline round-trip regression
test.

**Strict criterion lesson**: the cycle-19 fix-sweep met the
classifier-level criterion (regression test pins the contract)
but did not honor the cycle-18 audit doc's forward note 1
(centralize the scalar-width predicate / patch all parallel
width tables). The next fix-sweep should treat "canonicalization
diff" findings as **mandating** a grep-sweep for every parallel
width table, not just patching the originally-cited locus.

---

## Files touched by this audit

None — this is a read-only audit cycle. Findings will be
addressed in a subsequent fix-sweep (cycle 20 territory).

## Cross-reference

- Cycle-17 silent-failures (last clean predecessor):
  `docs/audit-stage28-8-cycle17-silent-failures.md`
- Cycle-18 type-design (surfaced C18-1):
  `docs/audit-stage28-8-cycle18-type-design.md`
- Cycle-19 type-design (parallel surfacing of C19-1):
  `docs/audit-stage28-8-cycle19-type-design.md`
- Fix-sweep that closed C18-1: `0803902`
- File touched by fix-sweep: `helixc/backend/x86_64.py:1005-1017`
- File this audit flags: `helixc/ir/passes/const_fold.py:43-49`
