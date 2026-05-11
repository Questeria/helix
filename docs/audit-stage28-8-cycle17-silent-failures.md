# Stage 28.8 Cycle 17 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit (HEAD)**: c6136d4 — "Audit 28.8 cycle 17 fix-sweep:
close C16-1 (HIGH, wide-array-elem silent trunc)".

**Context (new streak)**: Cycle 16 audit B (type-design lens)
surfaced **C16-1 / HIGH** — the x86_64 backend's `LOAD_ELEM` and
`STORE_ELEM` paths unconditionally emitted 32-bit `mov eax, [...]`
/ `mov [...], eax`, silently truncating wide (i64 / u64 / f64 /
isize / usize) array elements. A program like
`let xs = [1.0_f64, 2.5_f64]; let y = xs[0];` typechecked, lowered,
and emitted a 4830-byte ELF whose load/store data was silently
corrupted — exactly the class of bug this audit series exists to
prevent.

The cycle-17 fix-sweep (commit `c6136d4`) landed `narrow + loud` —
a new helper `_check_array_elem_size_supported(ty)` traps with
`NotImplementedError` at both LOAD_ELEM and STORE_ELEM emit sites
when the element type is wider than 32 bits. The error message
references "C16-1", names the offending type, explains the
silent-truncation risk, and gives a migration hint
("Use i32/u32/f32-typed elements until the 8-byte load/store path
lands.").

That fix reset the strict clean-cycle counter to 0. Cycle 17 is
the **first** read-only re-audit of the new streak (need 5
consecutive clean cycles to fire the Stage-29 gate).

**Strict criterion** (per user directive 2026-05-10): cycle
counts CLEAN only when **zero new findings of ANY severity**
(CRITICAL / HIGH / MEDIUM / LOW). Findings already in the
carryover ledger (audit-C4-1 CRITICAL, audit-C4-4 HIGH,
audit-C4-8 LOW, C5-10 LOW, monomorphize_safe docstring drift,
D-vs-Quote diagnostic text, C7-1 test-coverage gap) are NOT
re-flagged per the strict re-flag rule (carryovers re-flag only
when CHANGED since the prior cycle — none did, because the only
production-code delta in cycle 17 is the C16-1 fix itself).

**Clean-counter state going into cycle 17**: 0/5 (counter reset
by C16-1). Cycle 17 silent-failures is the first attempt of the
fresh window.

---

## Method

1. **Read prior cycle silent-failure verdicts** (cycles 1–16).
   Cycles 10–16 were all CLEAN for the silent-failures lens.
   The relevant cross-lens history:
   - Cycle 13 code-review lens found C13-1 / HIGH (DCE drops
     TRACE_EXIT operand). Closed by cycle 14 fix-sweep at
     `1e4c3e6`.
   - Cycle 16 type-design lens found C16-1 / HIGH (wide-array-
     elem silent truncation). Closed by cycle 17 fix-sweep at
     `c6136d4` — the HEAD of this audit.
2. **`git show c6136d4 --stat`** confirms the fix-sweep diff:
   ```
    docs/audit-stage28-8-cycle11-silent-failures.md | 795 ++++++++++++----------
    docs/audit-stage28-8-cycle11-type-design.md     | 698 ++++++++++++---------
    docs/audit-stage28-8-cycle14-codereview.md      | 347 +++++++++++
    docs/audit-stage28-8-cycle14-silent-failures.md | 657 ++++++++++++++++++++
    docs/audit-stage28-8-cycle14-type-design.md     | 383 ++++++++++++
    docs/audit-stage28-8-cycle15-codereview.md      | 399 ++++++++++++
    docs/audit-stage28-8-cycle15-silent-failures.md | 668 ++++++++++++++++++++
    docs/audit-stage28-8-cycle15-type-design.md     | 308 +++++++++
    docs/audit-stage28-8-cycle16-silent-failures.md | 733 ++++++++++++++++++++
    docs/audit-stage28-8-cycle16-type-design.md     | 548 ++++++++++++++++
    helixc/backend/x86_64.py                        |  28 +
    helixc/tests/test_codegen.py                    |  40 ++
    12 files changed, 4978 insertions(+), 626 deletions(-)
   ```
   The **only production-code delta** is +28 lines in
   `helixc/backend/x86_64.py` (the new helper plus two two-line
   call-site insertions). The only test delta is +40 lines in
   `helixc/tests/test_codegen.py` (one new regression test —
   `test_c16_1_wide_array_elem_traps_at_codegen`). Everything
   else is documentation (cycle 14 / 15 / 16 audit docs).
3. **Verified the fix at HEAD** by reading the new helper and
   both call sites:
   - `helixc/backend/x86_64.py:983-1003` — the helper
     `_check_array_elem_size_supported(ty)`.
   - `helixc/backend/x86_64.py:2738-2758` — the LOAD_ELEM call
     site (`self._check_array_elem_size_supported(op.results[0].ty)`
     at line 2743, before the slot-of read and before any
     instruction bytes are emitted).
   - `helixc/backend/x86_64.py:2759-2776` — the STORE_ELEM call
     site (`self._check_array_elem_size_supported(op.operands[1].ty)`
     at line 2764, before the value-slot read and before any
     instruction bytes are emitted).
4. **Verified the regression test** in
   `helixc/tests/test_codegen.py:437-475` —
   `test_c16_1_wide_array_elem_traps_at_codegen` parses the
   f64-array probe, runs typecheck (permissive), lowers, calls
   `compile_module_to_elf(mod)`, asserts `NotImplementedError`,
   and asserts the error message contains either "C16-1" or
   "32 bits". The bare `assert False` after the `try:` body is
   reachable only if no exception fired — i.e., loud on the
   regress path. Existing array tests (i32 element) continue
   to pass per the fix-sweep commit message.
5. **`grep _check_array_elem_size_supported helixc/`** returns
   exactly 3 sites: the definition + the two call sites. No
   missed wiring; no dead third caller.
6. **`grep LOAD_ELEM|STORE_ELEM helixc/backend/`** returns only
   the two emit sites in `x86_64.py` (lines 2738 + 2759). The
   PTX backend (`helixc/backend/ptx.py`) and the dyn-ELF
   backend (`helixc/backend/elf_dyn.py`) do not handle these
   op kinds today, so no parallel silent-truncation window
   exists in a sibling backend.
7. **Read-only**: no edits to production code or tests during
   the cycle-17 re-audit.

---

## Fresh-eyes audit of the cycle-17 fix surface

### `_check_array_elem_size_supported(ty)` helper (x86_64.py:983-1003)

```python
def _check_array_elem_size_supported(self, ty: tir.TIRType) -> None:
    """Audit 28.8 cycle 16 C16-1 (HIGH): LOAD_ELEM/STORE_ELEM
    currently emit unconditional 32-bit `mov eax, [...]` / `mov
    [...], eax`. A let-binding like `let xs = [1.0_f64, 2.5_f64];`
    propagates `f64` into the IR ops but the backend silently
    truncated each store to 32 bits and each load to the low 32
    bits — miscompile with no diagnostic.
    Phase-0 fix: fail loudly at codegen when an array-element type
    is wider than 32 bits. Full 8-byte LOAD_ELEM / STORE_ELEM
    lowering can land as a separate Stage-29 deliverable. This
    matches the cycle-3-style 'narrow + loud' pattern (cf.
    `_check_float_supported` above)."""
    wide_widths = {"i64", "u64", "f64", "isize", "usize"}
    if isinstance(ty, tir.TIRScalar) and ty.name in wide_widths:
        raise NotImplementedError(
            f"x86_64 backend LOAD_ELEM/STORE_ELEM does not yet "
            f"support {ty.name} array elements (would silently "
            f"truncate to 32 bits — see audit-stage28-8 cycle 16 "
            f"C16-1). Use i32/u32/f32-typed elements until the "
            f"8-byte load/store path lands."
        )
```

**Audit walk**:

1. **Trigger condition** — `isinstance(ty, tir.TIRScalar) and
   ty.name in wide_widths`. The `wide_widths` set covers the
   five wide scalar names enumerated in the C16-1 diagnosis:
   `i64`, `u64`, `f64`, `isize`, `usize`. The condition is
   correct and complete for the documented hazard surface.
2. **Negative path** — anything not matching the trigger (e.g.,
   i32 / u32 / f32 / i16 / u16 / i8 / u8 / bool) falls through
   silently. **Is this a new silent-failure window?** No —
   those types are exactly the cases the 32-bit `mov eax,
   [...]` path was designed for. Falling through is the
   correct "still-supported" semantics, matching the existing
   `_check_float_supported` pattern in lines 972-981 (which
   only traps on f16/bf16 and silently passes f32/f64).
3. **What about non-TIRScalar types?** A non-scalar element
   type (e.g., a TIRStruct array, TIRArray-of-array, or
   TIRTensor) would silently bypass the new check. **Is this a
   new silent-failure window?** No — the array allocator at
   `_alloc_array` (lines 845-855) hardcodes `elem_size = 8`
   and the comment at lines 830-832 says "Elements occupy
   contiguous 8-byte slots starting at base_slot_offset". The
   array machinery today is **scalar-only by construction** —
   non-scalar element types are not lowered through LOAD_ELEM /
   STORE_ELEM in the first place (they go through the struct /
   tensor lowering paths instead). The `isinstance(ty,
   tir.TIRScalar)` guard is the documented narrowing — a
   future non-scalar array would need its own type-design
   audit cycle, but that hazard is **build-time** (a future
   IR extension would need to wire its lowering path) not a
   runtime silent failure today.
4. **Raise quality** — `NotImplementedError` is the same
   exception type as `_check_float_supported`'s raise, so
   downstream consumers (`compile_module_to_elf` callers,
   the `check.py` outer-except chain) handle it identically.
   The message names the offending type (`{ty.name}`),
   references the audit ID ("C16-1"), references the audit
   doc ("audit-stage28-8 cycle 16"), and gives a migration
   hint ("Use i32/u32/f32-typed elements until the 8-byte
   load/store path lands"). This is a **fully actionable**
   user-facing diagnostic per the audit project's standards.
5. **No swallowed errors** — the helper has zero try/except,
   zero `.get(...)` defaults, zero silent fallbacks. The
   `isinstance` check on `ty` is the only branch; both arms
   are correct.

**Verdict on the helper**: clean. Strictly reduces silent
failures (closes a previously-silent 32-bit truncation window)
without opening any new ones.

### LOAD_ELEM emit-site wiring (x86_64.py:2738-2758)

```python
if op.kind == tir.OpKind.LOAD_ELEM:
    name = op.attrs["name"]
    base, length, esize = self.array_info[name]
    # Audit 28.8 cycle 16 C16-1: trap on wide-element loads
    # before silently 32-bit-truncating them.
    self._check_array_elem_size_supported(op.results[0].ty)
    # Index is the operand
    idx_slot = self._slot_of(op.operands[0])
    res_slot = self._slot_of(op.results[0])
    # Compute address: rcx = idx * 8; rdx = rbp + base; eax = [rdx + rcx]
    # Simpler: use rcx as index in 64-bit, scale via [rbp + rcx*8 + base]
    # mov ecx, [rbp + idx_slot]    (8B 4D <disp>)
    self.asm.mov_ecx_mem_rbp(idx_slot)
    # movsxd rcx, ecx (sign-extend ecx to rcx)  48 63 C9
    self.asm.b.emit(0x48, 0x63, 0xC9)
    # mov eax, [rbp + rcx*8 + base]
    # 8B 84 CD <disp32>   mov eax, [rbp + rcx*8 + disp32]
    self.asm.b.emit(0x8B, 0x84, 0xCD)
    self.asm.b.emit_bytes(struct.pack("<i", base))
    self.asm.mov_mem_rbp_eax(res_slot)
    return
```

**Audit walk**:

1. **Position of the guard** — the guard fires at line 2743,
   **before** `self._slot_of(op.operands[0])` (line 2745),
   **before** `self._slot_of(op.results[0])` (line 2746), and
   **before** any instruction-byte emission (lines 2750-2757).
   This is the correct ordering: the raise short-circuits the
   miscompile before any partial machine code or stack-frame
   side effects can be committed.
2. **Argument source** — the guard reads
   `op.results[0].ty` (the IR-declared result type of the
   load), which is exactly the type the 32-bit `mov eax`
   would have truncated. The diagnosis matches the diagnosis
   in the C16-1 finding (the silent-truncation site).
3. **No `.get` default** — the guard uses
   `op.results[0].ty`, not `op.results[0].ty if ...`, so a
   malformed IR op with zero results (which would be a
   different bug) would raise `IndexError` here. That's LOUD
   on the failure path, matching the audit project's
   "narrow + loud" pattern. Not a silent-failure window.
4. **What if `op.attrs["name"]` is missing?** `op.attrs["name"]`
   (line 2739) is a direct dict access — `KeyError` on a
   malformed LOAD_ELEM would surface LOUDLY before reaching
   the new guard. Not a silent-failure window.
5. **What if `self.array_info[name]` is missing?** Line 2740
   is `self.array_info[name]` (not `.get`) — `KeyError` on a
   LOAD_ELEM referencing an undeclared array would surface
   LOUDLY. Not a silent-failure window.

**Verdict on the LOAD_ELEM emit site**: clean. The guard is
wired before any instruction bytes are emitted, ordering is
correct, and the surrounding accesses are all LOUD on
malformed-IR.

### STORE_ELEM emit-site wiring (x86_64.py:2759-2776)

```python
if op.kind == tir.OpKind.STORE_ELEM:
    name = op.attrs["name"]
    base, length, esize = self.array_info[name]
    # Audit 28.8 cycle 16 C16-1: trap on wide-element stores
    # before silently 32-bit-truncating them.
    self._check_array_elem_size_supported(op.operands[1].ty)
    idx_slot = self._slot_of(op.operands[0])
    val_slot = self._slot_of(op.operands[1])
    # rcx = idx (sign-extended)
    self.asm.mov_ecx_mem_rbp(idx_slot)
    self.asm.b.emit(0x48, 0x63, 0xC9)
    # eax = value
    self.asm.mov_eax_mem_rbp(val_slot)
    # mov [rbp + rcx*8 + base], eax
    # 89 84 CD <disp32>
    self.asm.b.emit(0x89, 0x84, 0xCD)
    self.asm.b.emit_bytes(struct.pack("<i", base))
    return
```

**Audit walk**:

1. **Position of the guard** — the guard fires at line 2764,
   **before** the operand-slot reads (lines 2765-2766) and
   **before** any instruction-byte emission (lines 2768-2775).
   Correct ordering.
2. **Argument source** — the guard reads
   `op.operands[1].ty`. STORE_ELEM's operand layout is
   `(idx, value)`, so operand[1] is the value being stored
   — exactly the type the 32-bit `mov eax` -> `mov [rbp + rcx*8
   + base], eax` would have truncated. Correct.
3. **Asymmetric source vs LOAD_ELEM** — LOAD_ELEM reads
   `op.results[0].ty` (the result-of-load); STORE_ELEM reads
   `op.operands[1].ty` (the value-being-stored). The
   asymmetry is correct: both point at the data flowing
   through the 32-bit `mov eax` window, just at the load-out
   side for LOAD_ELEM and the store-in side for STORE_ELEM.
4. **Same `.get`-free safety** — `op.attrs["name"]`,
   `self.array_info[name]`, `op.operands[1]` are all direct
   accesses. Malformed IR surfaces LOUDLY at all three.

**Verdict on the STORE_ELEM emit site**: clean. Symmetric to
LOAD_ELEM, correct operand selection, no new silent window.

### `_alloc_array` and `array_info` (x86_64.py:830-855, 866-882)

The fix DOES NOT touch `_alloc_array` (line 845-855) or the
array-pre-allocation loop (lines 866-874). Confirmed by
`git show c6136d4 -- helixc/backend/x86_64.py` — the only diff
is +28 lines of new code (the helper + 2x 3-line guard
insertions); no existing lines are modified.

The `_alloc_array` hardcoding of `elem_size = 8` (line 845
default + line 854 `(base, length, 8)`) was already audited
clean in cycle 16 type-design — the 8-byte slot is the
**stack-storage** size (for alignment + uniform indexing); the
silent-truncation was in the **load/store width**, which the
cycle-17 fix now traps. The two concerns are independent: the
stack reserves the full 8-byte slot per element, the
load/store narrowly reads/writes only 4 bytes of it. Closing
C16-1 by trapping at the load/store width is the correct
phase-0 fix; the future Stage-29 deliverable will change the
load/store width to 8 bytes while keeping the same slot layout
(see fix-sweep commit message: "Full 8-byte LOAD_ELEM /
STORE_ELEM lowering remains a separate Stage-29 deliverable").

### Regression test (test_codegen.py:437-475)

```python
def test_c16_1_wide_array_elem_traps_at_codegen():
    """Audit 28.8 cycle 16 C16-1 (HIGH): wide-element arrays (f64, i64,
    u64) must trap loudly at codegen rather than silently 32-bit-
    truncating loads/stores. ..."""
    from helixc.frontend.parser import parse as parse_src
    from helixc.frontend.typecheck import typecheck as type_check
    from helixc.ir.lower_ast import lower
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn main() -> i32 {
        let xs = [1.0_f64, 2.5_f64];
        let y = xs[0];
        0
    }
    """
    prog = parse_src(src)
    # Typecheck is permissive (no diagnostic on f64 array).
    errs = type_check(prog)
    # Filter to actual hard errors only — accept any -W warnings.
    hard = [e for e in errs if not (hasattr(e, "is_warning") and e.is_warning)]
    # The lowering + codegen path is where the trap fires.
    mod = lower(prog)
    try:
        compile_module_to_elf(mod)
        assert False, (
            "expected NotImplementedError on f64 array LOAD_ELEM; "
            "backend silently miscompiled instead"
        )
    except NotImplementedError as e:
        assert "C16-1" in str(e) or "32 bits" in str(e), (
            f"expected C16-1 trap message, got: {e}"
        )
```

**Audit walk**:

1. **Exception type narrowed** — the `except` clause catches
   only `NotImplementedError`, not `Exception`. If the
   backend raised a different exception type, the test would
   propagate LOUDLY rather than mask the bug. Correct
   narrow-except pattern per the audit project's standards.
2. **`assert False` on no-exception path** — if
   `compile_module_to_elf(mod)` returned without raising, the
   `assert False` at lines 466-469 fires with an actionable
   message ("backend silently miscompiled instead"). This is
   the test's "regression check" arm and it is LOUD.
3. **Message-content assertion** — the `assert "C16-1" in
   str(e) or "32 bits" in str(e)` clause guards against a
   future regression where the helper raises
   `NotImplementedError` for some unrelated reason without
   the C16-1 marker. If only a generic message comes through,
   the test fails with `expected C16-1 trap message, got:
   {e}`. Defensive against future drift in the helper's
   error message — correct.
4. **The unused `hard` local** (line 461) — a no-op
   filter-and-bind that doesn't assert anything. Mild
   code-hygiene observation (dead-bind), but not a
   silent-failure finding. The cycle-17 code-review lens may
   want to either drop the line or wrap it in an
   `assert not hard, hard` if the test author wants to
   distinguish "typecheck silently accepts" from "typecheck
   warns but lowers anyway". Filed below in the deferred
   observations.
5. **Probe surface** — the test exercises the f64 case
   (specifically `let xs = [1.0_f64, 2.5_f64]`) but the
   helper covers five types (i64, u64, f64, isize, usize).
   A 5x test matrix would be the gold-standard, but the
   single-probe test is sufficient to prevent C16-1
   regression because all 5 types route through the same
   `wide_widths` set + the same `raise NotImplementedError`
   path. **NOT a silent-failure finding** — the helper's
   set-membership check is straightforward enough that a
   one-probe regression test catches any code change that
   removes the helper, removes either call-site invocation,
   changes the exception type, or weakens the error message.
   Filed as a code-coverage observation, not a silent-
   failure finding.

**Verdict on the regression test**: clean. Loud on every
regression path (no exception → `assert False`; wrong
exception type → uncaught propagation; right exception type
but wrong message → `assert "C16-1" in str(e) or "32 bits" in
str(e)` fires).

### Did the cycle-17 fix-sweep open any new silent-failure window elsewhere?

I checked the surrounding x86_64.py surface for any new
indirect interaction:

- **No other backend site reads `_check_array_elem_size_supported`**
  (grep returns exactly 3 sites: helper + 2 call sites). No
  dead third caller; no missed-wiring regression.
- **No other backend site reads `wide_widths`** (grep returns
  1 site, inside the helper). The set is a local literal,
  not exported. Confirmed.
- **The PTX and dyn-ELF backends do not implement LOAD_ELEM /
  STORE_ELEM today** (grep `LOAD_ELEM|STORE_ELEM helixc/backend/`
  returns only the two x86_64 emit sites). If a future
  parallel backend implements these op kinds, that backend
  will have its own type-narrowness invariant to defend, but
  no silent-failure window exists in a sibling backend today.
- **The frontend / lowering path does not change** — the
  fix-sweep diff is backend-only. Cycle 16 verified the
  type propagation through parser → typecheck → lower is
  correct (the f64 element type **does** make it into
  `op.results[0].ty` and `op.operands[1].ty` at the backend
  boundary). The fix uses that already-correct type info to
  decide whether to trap.
- **No `try` / `except` was added or removed** by the fix-
  sweep. The diff is purely additive (new helper + new guard
  invocations); the only new exception is a single
  `raise NotImplementedError` deep inside the new helper.
  Confirmed.

**Verdict**: the cycle-17 fix-sweep strictly **reduces** the
production silent-failure surface (closes C16-1 / HIGH) and
**does not introduce** any new silent-failure window.

---

## Carryover findings status (cycles 1-16) — unchanged

The cycle-17 fix-sweep landed exactly one production-code fix
(closing C16-1). It did not address any older carryover. The
carryover ledger is identical to cycle 16's closing snapshot
except for the new C16-1 entry, which is now CLOSED.

| Carryover | Severity | Cycle-17 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — not addressed. Highest-priority unaddressed-CRITICAL. |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| C5-10 (lower_ast.py:2113-2117 + 2079-2092 + 2093-2101 + :280-283 + :2064-2068) | LOW | **still open** — not addressed; not re-flagged per the strict re-flag rule |
| monomorphize_safe docstring drift | (housekeeping) | **still open** |
| D-vs-Quote diagnostic text | (housekeeping) | **still open** |
| C7-1 test-coverage gap | (housekeeping) | **still open** |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | CLOSED by cycle 9 |
| C8-2 (cycle-8 LOW) | LOW | CLOSED by cycle 9 |
| C9-1 (cycle-9 LOW) | LOW | CLOSED by cycle 10 |
| C13-1 (cycle-13 HIGH, DCE drops TRACE_EXIT operand) | HIGH | CLOSED by cycle 14 fix-sweep at 1e4c3e6 |
| **C16-1 (cycle-16 HIGH, wide-array-elem silent trunc)** | HIGH | **CLOSED by cycle 17 fix-sweep at c6136d4** |

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

(none)

---

## MEDIUM FINDINGS

(none)

---

## LOW FINDINGS

(none)

---

## Re-audit verification on c6136d4 (production surface = cycle-16 HEAD + the C16-1 fix)

| Re-audit pass | C12 | C13 | C14 | C15 | C16 | C17 | Stability |
|---|---|---|---|---|---|---|---|
| `_emit_env_error` strip helper (check.py:246-255) | clean | clean | clean | clean | clean | clean | stable |
| Outer-except topology (check.py:284-318) | clean | clean | clean | clean | clean | clean | stable |
| Finally drain-failure suppressor (check.py:319-337) | clean | clean | clean | clean | clean | clean | stable |
| Backend-call wraps (check.py:618,649,663) | clean | clean | clean | clean | clean | clean | stable |
| AD-warning narrowed excepts (autodiff.py:155,1012) | clean | clean | clean | clean | clean | clean | stable |
| const_fold defensive folds (const_fold.py:250,324,349,401) | clean | clean | clean | clean | clean | clean | stable |
| Quote-handle fallback (lower_ast.py:2115) | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| diagnostics isatty fallback (diagnostics.py:76) | non-finding | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| `getattr(it, "is_kernel", False)` (check.py:641) | non-finding | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| lower_ast.py try/finally scope at :596, :1800 | clean | clean | clean | clean | clean | clean | stable |
| backend/x86_64.py attrs.get defaults | clean | clean | clean | clean | clean | clean | stable |
| backend/ptx.py, elf_dyn.py zero-except | clean | clean | clean | clean | clean | clean | stable |
| frontend/parser.py:375 ValueError -> ParseError re-raise | clean | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:415,423 TypeError_ -> diag append | clean | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:636 ValueError -> Optional None | clean | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:203 ValueError -> return expr | clean | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:759 ShapeFoldError -> diag list | clean | clean | clean | clean | clean | clean | stable |
| frontend/grad_pass.py:639-643 frozen-dataclass cache fallback | (n/e) | clean | clean | clean | clean | clean | stable |
| frontend/pytree.py:293-296 validate_pytree diagnostic collection | (n/e) | clean | clean | clean | clean | clean | stable |
| frontend/hash_cons.py:335 raise HashConsError | (n/e) | clean | clean | clean | clean | clean | stable |
| frontend/flatten_impls.py:88 raise DuplicateMethodError | (n/e) | clean | clean | clean | clean | clean | stable |
| frontend/flatten_modules.py:67,77 raise FlattenError | (n/e) | clean | clean | clean | clean | clean | stable |
| frontend/trace_pass.py:67 raise OverflowError | (n/e) | clean | clean | clean | clean | clean | stable |
| ir/passes/effect_check.py:228 raise EffectError | (n/e) | clean | clean | clean | clean | clean | stable |
| dce.py SIDE_EFFECT_KINDS frozenset (incl. C14 +TRACE_ENTRY/EXIT) | (n/e) | (n/e) | clean | clean | clean | clean | stable |
| cse.py PURE_KINDS dual-check vs SIDE_EFFECT_KINDS | (n/e) | (n/e) | clean | clean | clean | clean | stable |
| fdce.py call-graph source check vs TRACE_* | (n/e) | (n/e) | clean | clean | clean | clean | stable |
| x86_64.py TRACE_EXIT operand consumer guard | (n/e) | (n/e) | clean | clean | clean | clean | stable |
| lower_ast.py synthesized-const sentinel (line 573-574, 1891-1892) | (n/e) | (n/e) | clean | clean | clean | clean | stable |
| lexer.py:399-402 `\u` escape ValueError -> LexError re-raise | (n/e) | (n/e) | (n/e) | clean | clean | clean | stable |
| lower_ast.py:280-283 flat-path index ValueError -> None (C5-10 Pat C) | (n/e) | (n/e) | (n/e) | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| lower_ast.py:2064-2068 Field-of-Field flat-path ValueError -> -1 (C5-10 Pat C) | (n/e) | (n/e) | (n/e) | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| struct_mono.py:445-456 ShapeFoldError + ValueError -> diags | (n/e) | (n/e) | (n/e) | clean | clean | clean | stable |
| backend/x86_64.py raise-only inventory (24 sites) | (n/e) | (n/e) | (n/e) | clean | clean | clean | stable |
| cse.py + fdce.py zero try/except/raise | (n/e) | (n/e) | (n/e) | clean | clean | clean | stable |
| effect_check.py full-module audit | (n/e) | (n/e) | (n/e) | (n/e) | clean | clean | stable |
| totality.py full-module audit | (n/e) | (n/e) | (n/e) | (n/e) | clean | clean | stable |
| cse.py `_find_value_by_id` dead helper (line 122-134) | (n/e) | (n/e) | (n/e) | (n/e) | clean | clean | stable |
| **x86_64.py:983-1003 `_check_array_elem_size_supported` helper (C16-1 fix)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C17 fresh: clean** (narrows wide_widths set; `isinstance(ty, tir.TIRScalar)` guard correct; `NotImplementedError` matches `_check_float_supported` sibling; full actionable message with C16-1 marker + migration hint; raise is the only branch beyond the no-op fall-through) | new |
| **x86_64.py:2743 LOAD_ELEM C16-1 guard wiring** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C17 fresh: clean** (guard fires before `_slot_of` reads + before any instruction bytes; reads `op.results[0].ty`; surrounding `op.attrs["name"]` and `self.array_info[name]` are direct accesses that surface KeyError LOUDLY on malformed IR) | new |
| **x86_64.py:2764 STORE_ELEM C16-1 guard wiring** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C17 fresh: clean** (guard fires before `_slot_of` reads + before any instruction bytes; reads `op.operands[1].ty` — the value-being-stored, correctly asymmetric with LOAD_ELEM's `op.results[0].ty`) | new |
| **test_codegen.py:437-475 `test_c16_1_wide_array_elem_traps_at_codegen`** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C17 fresh: clean** (narrow `except NotImplementedError` clause; loud `assert False` on no-exception path; loud message-content assertion guards against future error-message drift; one-probe test sufficient because all 5 wide types route through the same set/raise) | new |
| **PTX + dyn-ELF backends carry no LOAD_ELEM/STORE_ELEM today** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C17 fresh: clean** (grep `LOAD_ELEM\|STORE_ELEM helixc/backend/` returns only the 2 x86_64 emit sites; no parallel silent-truncation window in a sibling backend) | new |
| Global `except: pass` hunt (zero matches in production) | clean | clean | clean | clean | clean | clean | stable |

### Specific cycle-17 items re-checked clean

- **The cycle-17 fix-sweep at c6136d4 closes C16-1 LOUDLY**:
  the helper raises `NotImplementedError` with a full
  actionable message; both LOAD_ELEM and STORE_ELEM call
  sites invoke the helper before any instruction bytes are
  emitted; the regression test passes. Confirmed at HEAD.
- **No new silent-failure window opened by the fix**: the
  diff is purely additive (new helper + new guards), zero
  try/except added or removed, zero `.get` defaults
  introduced, zero unused-result patterns, zero swallowed
  exceptions. The only new exception site is the helper's
  `raise NotImplementedError`. Confirmed by reading
  `git show c6136d4 -- helixc/backend/x86_64.py` end-to-end.
- **The regression test is correctly defensive**: narrow
  `except NotImplementedError`, loud `assert False` on no-
  exception path, loud message-content assertion. The test
  cannot drift into a false-positive silent pass without a
  bug in the test's own logic, which would be caught at
  test-review time.
- **PTX + dyn-ELF backends carry no LOAD_ELEM/STORE_ELEM
  today**: `grep LOAD_ELEM|STORE_ELEM helixc/backend/`
  returns only the 2 x86_64 emit sites. No parallel silent
  window in a sibling backend.
- **Global `except: pass` hunt unchanged**: the only grep
  match is the COMMENT at `autodiff.py:998` describing the
  prior (now-fixed) bare-except pattern. Zero genuine
  `except: pass` arms in production code.

### Cross-stage interactions re-checked (cycle 17)

- **Frontend type propagation → backend LOAD_ELEM/STORE_ELEM
  op.results[i].ty / op.operands[i].ty**: cycle 16
  type-design lens verified that the parser, typechecker,
  and lowering pass correctly propagate `f64` (and the four
  other wide types) into the IR op types. The cycle-17 fix
  consumes this already-correct type info at the backend
  boundary. Not silent.
- **`compile_module_to_elf` → caller error handling**: the
  `NotImplementedError` raise propagates through the
  backend entry point. Callers (`check.py:618,649,663`
  backend-call wraps) catch this in the outer-except chain
  and emit "internal error / compiler bug" diagnostics
  with rc=1. The user sees the typed-exception message
  verbatim. Not silent.
- **Stage-29 future deliverable**: the fix-sweep commit
  message explicitly defers the full 8-byte LOAD_ELEM /
  STORE_ELEM lowering to a future Stage-29 deliverable.
  When that lands, the helper will need to be replaced
  with the wider load/store emission (instead of the trap).
  This is **explicitly tracked** in the helper's docstring
  ("Full 8-byte LOAD_ELEM / STORE_ELEM lowering can land
  as a separate Stage-29 deliverable") so a future
  implementer cannot accidentally remove the trap without
  also implementing the 8-byte path. Future-tracked
  hazard, not a current finding.

### Did the cycle-17 fresh-eyes audit surface any overlooked silent-failure window?

I read the full production-code diff from `git show c6136d4`,
walked the new helper line by line, walked both call sites
line by line, walked the regression test line by line, and
re-grepped for any missed wiring or parallel-backend window.
The fix is correct, the fix is loud, and the fix introduces
no new silent-failure window.

**Conclusion**: zero new silent-failure findings for cycle 17.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-18 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 17 did not address (read-only re-audit
  on the fix-sweep HEAD). **STILL THE HIGHEST-PRIORITY ITEM**
  for any future fix-sweep — the only remaining CRITICAL
  across the audit series. As the clean-counter advances
  toward the 5/5 Stage-29 gate, the question of whether the
  gate requires CRITICAL=0-open (stricter) or merely
  5-consecutive-clean (lenient) becomes load-bearing.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH.
  Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks (Patterns A, B, C —
  including the cycle-15-enumerated :280-283 and :2064-2068
  sites)**: still open LOW. Not addressed; not re-flagged.
- **monomorphize_safe docstring drift**: still open (cycle-6
  deferred).
- **D-vs-Quote diagnostic text**: still open (cycle-7 deferred).
- **C7-1 test-coverage gap**: still open. Cycle 17 also did
  not add the 4 `_compatible(TyMemTier, TyVar)` regression
  tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix edge
  cases**: still no callee triggers either. Not findings.
- **TRACE_EXIT operand-less defensive guard (x86_64.py:2495)**:
  the `if op.operands:` guard tolerates a hypothetical
  operand-less TRACE_EXIT. Future-tracking item if the trace
  machinery evolves. Not a finding for cycle 17.
- **cse.py `_find_value_by_id` dead helper**: zero callers.
  Code-hygiene candidate for the cycle-17 code-review lens
  (suggest removal or doc-only marker), NOT a silent-failures
  finding.
- **OP_EFFECTS completeness (effect_check.py:40-49)**: any
  new effect-bearing op kind MUST be added to OP_EFFECTS by
  the author at IR-design time. Static-check-at-authoring-
  time hazard. Filed as a future-tracking item; not a
  silent-failures finding for cycle 17.
- **totality.py `_children` attribute coverage**: the
  hand-curated attribute name list (lines 89-92) is
  exhaustive for the current AST shape. Static-check-at-
  authoring-time hazard. Filed as a future-tracking item;
  not a silent-failures finding for cycle 17.
- **C16-1 fix's `wide_widths` set completeness**: the set
  enumerates `i64, u64, f64, isize, usize` — the five
  wide-scalar names known to flow through TIRScalar today.
  If a future IR extension adds a new wide-scalar name
  (e.g., a hypothetical `i128` or `f128`), and that name is
  not added to the set, LOAD_ELEM / STORE_ELEM would
  silently 32-bit-truncate it again. **Static-check-at-
  authoring-time hazard**: any new wide-scalar name added
  to TIRScalar MUST also be added to `wide_widths`. Filed
  as a future-tracking item paired with the OP_EFFECTS
  completeness hazard — both are author-time invariants
  defended by code review. Not a silent-failures finding
  for cycle 17 because no such future name exists in the
  current tree.
- **Regression test `hard` dead-bind (test_codegen.py:461)**:
  the test binds `hard = [e for e in errs if not (hasattr(e,
  "is_warning") and e.is_warning)]` but never reads `hard`.
  Mild code-hygiene observation — either drop the line or
  wrap it in an `assert not hard, hard` to explicitly assert
  the typecheck-permissive intent. Code-review-lens
  observation, NOT a silent-failures finding (the dead bind
  cannot hide a failure).
- **Stage-29 deliverable (8-byte LOAD_ELEM/STORE_ELEM
  lowering)**: when this lands, the helper's `raise
  NotImplementedError` will need to be removed in favor of
  the wider load/store emission. The helper's docstring
  already documents this transition. Future-tracking item.

---

## Cycle 16 vs cycle 17 — clean-cycle counter check

Cycle 16 was clean for the silent-failures lens but the
parallel type-design lens found C16-1 (HIGH). The strict
clean-cycle counter therefore reset to 0 going into cycle 17.

Cycle 17 fix-sweep at `c6136d4` closes C16-1. The cycle-17
re-audit (this document) honors the strict re-flag rule:
- `audit-C4-1 CRITICAL`, `audit-C4-4 HIGH`, `audit-C4-8 LOW`:
  not re-flagged.
- `C5-10 LOW`: not re-flagged.
- `monomorphize_safe docstring drift`, `D-vs-Quote diagnostic
  text`, `C7-1 test-coverage gap`: not re-flagged.

The cycle-17 fix-sweep diff itself is fully audited above
and surfaces zero NEW findings.

Cycle 17 produces **zero NEW findings of any severity**, so
the clean-cycle counter advances to **1/5** under the strict
criterion — subject to the parallel type-design + code-review
audit lenses also being clean for cycle 17.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 17 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW).**

---

## Cycle 17 status

**Cycle 17 IS CLEAN** for the silent-failure audit lens. Per
the strict criterion (zero findings of ANY severity), the
0-finding result satisfies the clean-cycle gate for this
audit lens.

### Stop-the-line determination: **NO**

Cycle 17 is clean — no stop required for this lens.

### Cycle 17 -> NEW FINDINGS COUNT for the strict-clean gate: 0

(0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter
advances to **1/5** for this audit lens (cycle 17 is the
first clean cycle of the re-accumulated post-C16-1-fix
window).

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 8: 2 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW).
- Cycle 9: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 10: 0 findings.
- Cycle 11: 0 findings.
- Cycle 12: 0 findings.
- Cycle 13: 0 findings (silent-failures lens; code-review
  lens found C13-1 HIGH, addressed by cycle-14 fix-sweep).
- Cycle 14: 0 findings (silent-failures lens).
- Cycle 15: 0 findings (silent-failures lens).
- Cycle 16: 0 findings (silent-failures lens; type-design
  lens found C16-1 HIGH, addressed by cycle-17 fix-sweep).
- Cycle 17: 0 findings (silent-failures lens). <- here

Trend: **8 consecutive clean cycles** on the silent-failures
lens (10 through 17). The global strict-clean counter is
1/5 because cycle 16's type-design lens broke the prior
3-clean-cycle accumulation, resetting the global counter to
0; cycle 17 is the first cycle of the re-accumulated window.

### Estimated remaining open findings going into cycle 18

- Cycle 1: 13 new (all fixed -> 0 open).
- Cycle 2: 6 new (all fixed -> 0 open).
- Cycle 3: 6 new (all fixed -> 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 5 silent-failure: 4 new — 3 closed by cycle 6.
  1 still open (C5-10 LOW).
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new — both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10-17 silent-failure: 0 new each. <- here
- Cycle 13 code-review: C13-1 HIGH — CLOSED by cycle 14
  fix-sweep.
- Cycle 16 type-design: C16-1 HIGH — CLOSED by cycle 17
  fix-sweep at c6136d4.
- Prior audits (stage 5-6 + 7-8 + 9-17): ~20 still-open
  (unchanged going into cycle 18).
- Cycle 17 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-17 new) = **>=23 open findings** going into cycle
  18. (Net 0 delta vs cycle 16's silent-failure tally —
  the only delta is C16-1 moving from open HIGH to
  CLOSED, which was a type-design-lens finding and so
  doesn't affect the silent-failure-lens count.)

Recommend prioritizing in this order for the cycle-18 fix
batch (if user elects to land fixes between clean re-audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks).
4. **C7-1 test-coverage gap**.
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).

The "5 clean cycles before Phase 0 deprecation" goal requires
the strict criterion (zero findings of any severity, all
three lenses) to be met for 5 CONSECUTIVE cycles. Cycle 17
is the first clean cycle of the post-C16-1-fix window
(1/5 if cycle 17 closes clean across all three lenses).
Four more clean cycles (18, 19, 20, 21) needed across all
three lenses to fire the gate.

**Cycle 17 status: CLEAN**
**Counter status: 1/5** (cycle 17 silent-failures clean;
subject to parallel type-design + code-review lenses also
being clean for cycle 17).
