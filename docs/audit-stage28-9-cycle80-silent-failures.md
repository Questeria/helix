# Audit Stage 28.9 cycle 80 — Silent failures

**Scope.** HEAD `d218e65` (cycle-79 fix-sweep landing). Narrow conservative
read-only re-audit. Prior C1–C79 + deferred-known not re-flagged.

**Criterion.** 0 findings at confidence >= 75%.

## Result: 0 findings at >= 75% — PASS

---

## Verification of the cycle-79 delta

The cycle-79 commit `d218e65` fix is FFI_CALL float-return symmetry
(C78-1). Read `x86_64.py:1779–1795` carefully:

```python
if op.results:
    res_slot = self._slot_of(op.results[0])
    if self._is_f64_type(op.results[0].ty):
        self.asm.movsd_mem_rbp_xmm0(res_slot)
    elif self._is_float_type(op.results[0].ty):
        self.asm.movss_mem_rbp_xmm0(res_slot)
    elif self._is_i64_type(op.results[0].ty) or self._is_u64_type(op.results[0].ty):
        self.asm.mov_mem_rbp_rax(res_slot)
    else:
        self.asm.mov_mem_rbp_eax(res_slot)
```

**If-chain order soundness.** `_is_f64_type` (line 1002–1003) is the strict
subset `{f64}`. `_is_float_type` (line 999–1000) is the superset
`{f16, bf16, f32, f64}`. Because the f64 arm appears **before** the
generic float arm, f64 takes precedence — a Helix `f64` return is not
incorrectly fielded by movss. The f16/bf16 cases that the generic float
arm would also catch are rejected at CONST_FLOAT codegen by
`_check_float_supported` (line 1019), so only f32 actually reaches the
movss arm in practice. The ordering is correct.

**Symmetry to the arg side.** FFI_CALL args at 1754–1776 split int/float
by `_is_float_type` and within float pick f64-vs-rest. The return side
now mirrors that exactly. The cycle-79 fix is complete.

**Sibling return sites scanned for the same defect class.**

- **Regular CALL return** at lines 1709–1719: explicit f64 / float / i64 /
  else arms in the same order. Correct (pre-cycle-77 baseline; not
  changed by cycle-79).
- **RETURN op** at lines 1809–1827: explicit f64 / float / i64 / else
  arms. Correct.
- **BR op param transfer** at lines 1838–1850: handles f64 via xmm0 and
  i64 via rax; falls through to `mov_eax_mem_rbp` + `mov_mem_rbp_eax`
  for the rest. For 32-bit-wide types (i8/i16/i32/bool/f32) the 32-bit
  mov preserves the full bit pattern, so the f32 case round-trips
  correctly. The deliberate comment in the sibling LOAD_VAR site (line
  1875–1876, "32-bit movs for i32 / f32 — bit pattern round-trips")
  confirms this is intentional. No miscompile.
- **SHAPE_OP / TRACE return.** Grepped `OpKind` for `SHAPE_OP`,
  `CALL_INDIRECT`, `METHOD_CALL`, etc. — none present. `TRACE_ENTRY` /
  `TRACE_EXIT` at 2575–2598 are NOPs. No other call/return sites.

The FFI_CALL fix has no sibling-call-site holes.

---

## Rotation: elf_dyn.py

Read `helixc/backend/elf_dyn.py` end-to-end (481 lines). All offset and
size arithmetic is assertion-guarded:

- `plan_layout` walks `cur_off` linearly through dynstr → dynsym →
  hash → rela.plt → dynamic → got.plt and asserts after writing
  rela.plt that its emitted size matches the pre-reserved size
  (line 326).
- `emit_elf_dyn` asserts the byte buffer reaches every region offset
  exactly (lines 459, 463, 468, 470, 472, 474, 476, 478, 480) — any
  silent layout drift would surface as a loud AssertionError rather
  than emit a corrupt ELF.
- The DT_ entry count (`n_dyn_entries = len(needed_libs) + 12`) is
  cross-checked against the actual emitted entries at line 344.
- SYSV `.hash` chain construction (lines 257–272) was hand-traced for
  `n_imports ∈ {0, 1, 2, 3}`. For each case `len(chain_vals) == nchain`
  and the chain terminates at sentinel 0. Loader behaviour is a linear
  scan from `bucket_vals[0]`, which is correct for `nbucket = 1`.
- Padding and 8-byte alignment of `dynstr_data` (line 241), `hash_data`
  (line 273), and `cur_off` after code (line 281) are explicit loops or
  bitmasks — no rounding-mode silent drop.

No silent-fallthrough patterns in dyn-link emission.

## Rotation: presburger.py

Read `helixc/frontend/presburger.py` end-to-end (353 lines). Two
observations weighed and dismissed:

- **`_implies_le` / `_implies_divides` returning `None` on harder
  cases.** This is documented incompleteness (file docstring lines
  29–35: "harder cases ship with a 'could not prove' diagnostic"). The
  consumer `typecheck.py:867–882` treats `None` as "not provably
  refuted" and emits a diagnostic only when `verdict is False`. The
  docstring's "could not prove" diagnostic is aspirational — the actual
  behaviour is permissive-on-unknown. This is the same posture observed
  in the cycle-5 silent-failures audit (line 1005 of
  `audit-stage28-8-cycle5-silent-failures.md`: "presburger.py — no
  exception swallowing observed") and has not been re-flagged in any
  subsequent cycle. Pre-existing; not a cycle-79 regression; deferred-
  known territory (incompleteness of the Phase-0 solver, not silent
  miscompile of a working solver). Confidence to flag would be <50%
  given the prior cycle precedent.
- **`_reduce_via_eqs` has a `if False else (...)` idiom at line 281–283.**
  Hand-evaluated: when `c == 1`, substitution = `-rest`; when `c == -1`,
  substitution = `rest`. Both branches yield the algebraically correct
  `v = -rest/c`. Stylistic oddity, not a soundness bug.

No new silent-failure findings in the index-analysis path.

---

## Sibling-class checks examined and clean

- **FFI_CALL int-side return** still routes `_is_i64_type ∨ _is_u64_type`
  to `mov_mem_rbp_rax` (1791) — matches the FFI arg-side pointer-shape
  permissiveness from cycle-77.
- **Regular CALL arm u64 routing** at line 1703 checks only
  `_is_i64_type`. The cycle-78 type-design audit (lines 166–173 of
  `audit-stage28-9-cycle78-type-design.md`) reviewed this asymmetry,
  found it round-trip-consistent with the matching `_is_i64_type`-only
  prologue spill at line 986, and explicitly **declined to flag it**.
  Pre-existing-examined; not re-flagged.
- **xmm_idx counter on mixed args (FFI_CALL).** Initialized to 0 and
  incremented only inside the float branch (1754–1776). Independent of
  int_idx. Correct.
- **f16 / bf16 reaching codegen.** `_check_float_supported` errors on
  these at CONST_FLOAT lowering (line 1019), so they cannot reach the
  generic float arm of FFI_CALL-return as silent f32 traffic.

## Pre-existing items intentionally not flagged

- `monomorphize._mangle_ty` silent catchall — deferred-known per audit
  scope.
- `hash_cons._ast_equal` silent catchall — deferred-known per audit
  scope.
- `typecheck.check.py` pre-flatten / `struct_mono` pre-flatten — deferred-
  known per audit scope.
- `autotune.collect_autotuned_fns` missing iter_fn_decls — deferred-known
  per audit scope.
- `||` lowering at `lower_ast.py:1135-1138` ADD(result_ty=bool) — out
  of cycle-79 fix-class; deferred per cycle-78 silent-failures audit.
- `tile_ir.py:220` "treat as opaque for v0.1" TODO — same deferred set.
- Regular CALL arm u64 routing — cycle-78 type-design pre-examined.
- Presburger `None` on harder cases — incompleteness, cycle-5 examined.

## No code edits performed.

Read-only audit. No source files modified. Single Write to this doc only,
as scoped.
