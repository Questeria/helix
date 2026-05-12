# Audit Stage 28.9 cycle 85 — Silent failures

**Scope**: HEAD `fb80a4f` (Stage 28.9 only; Stage 28.10 commits explicitly out of scope).
**Mode**: STRICT READ-ONLY (Read/Grep/Glob/Bash only; one Write to this file; no Edit).
**Result**: **FAIL** — 1 finding at confidence >= 75%.

---

## Deferred-known items NOT re-flagged

Per scope:
- `helixc/ir/passes/monomorphize.py::_mangle_ty` silent catchall (deferred).
- `helixc/ir/passes/hash_cons.py::_ast_equal` silent catchall (deferred).
- `helixc/frontend/check.py` typecheck/struct_mono pre-flatten (deferred).
- `helixc/frontend/autotune.py::collect_autotuned_fns` missing `iter_fn_decls` (deferred).

C1–C84 prior findings and prior deferred-known items: not re-flagged.

---

## Cycle-84 fix verification (test-runner discovery)

`helixc/tests/test_ffi.py` — globals() discovery present in `__main__` block;
4 `def test_*` functions in file, all callable in `main_refs`. Cycle-84 patch
correctly replaces the hard-coded 3-test list (which had silently skipped
`test_c76_1_ffi_call_routes_f32_args_to_xmm0` since cycle 79). **Verified PASS.**

### Survey: other test runners with hard-coded test lists

Scanned all 37 `helixc/tests/test_*.py` files. 10 files do not use `globals()`
discovery (`test_autotune`, `test_cli`, `test_deprecated`, `test_diagnostics`,
`test_panic`, `test_provenance`, `test_pytree`, `test_struct_mono`, `test_trace`,
`test_unsafe`). All 10 use `pytest.main([__file__, ...])` in their `__main__`
block — pytest auto-collects every `def test_*` function in the file, equivalent
to (in fact more thorough than) the globals() pattern. **No silent miss.**

2 files (`test_ast_walker`, `test_codegen_determinism`) have no `__main__` block —
they only run under explicit pytest invocation. Not silently skipped: they don't
run via `python <test>.py`. Out-of-scope from cycle-84 regression class.

Cycle-84 fix scope is **complete and correct**.

---

## Findings

### C85-1 (HIGH conf 90) — const_fold SHL/SHR bound uses fixed `[0, 63]` instead of result-type bitwidth: silent miscompile for sub-i64 shifts where `bitwidth(res.ty) <= r < 64`

**File**: `helixc/ir/passes/const_fold.py:491-502`

```python
elif op.kind == tir.OpKind.SHL:
    if r < 0 or r >= 64:
        raise ShiftFoldError(...)
    v = l << r
elif op.kind == tir.OpKind.SHR:
    if r < 0 or r >= 64:
        raise ShiftFoldError(...)
    v = l >> r
```

The bound `r >= 64` is i64-correct but wrong for i8/i16/i32/u8/u16/u32 results.
For `1_i32 << 32_i32` the const_fold produces value 0 (Python `1 << 32 = 2^32`,
then `_wrap_int_to_type(2^32, i32) = 0`). The x86-64 backend's `shl_eax_cl`
relies on hardware `SHL r32, cl` semantics, which mask the count register to
the low 5 bits (Intel SDM Vol. 2B): `cl=32 & 0x1F = 0`, so runtime computes
`1 << 0 = 1`. const_fold says 0, hardware says 1 — **silent miscompile**.

**Reproduced** on HEAD `fb80a4f`:

```
fn main() -> i32 {
    let x: i32 = 1_i32 << 32_i32;
    x
}
```

After `fold_module`: `CONST_INT value=0`. Hardware runtime would emit 1.

Similar drift across i32 SHL/SHR for any compile-time `r` in `[32, 63]`, i16 for
`r in [16, 63]`, i8 for `r in [8, 63]`. (Negative `r` and `r >= 64` are correctly
trapped; the gap is the type-dependent middle window.) Surface Helix accepts
the construct — no frontend rejection of `i32 << 32`.

The shift-bound work landed in cycle 19 (audit-A C19-1) for the trap-17002
diagnostic and cycle 21 (audit-R C20-R1) for the swallow-by-`except Exception`
fix. Neither cycle made the bound result-type-aware. The regression test
`test_stage19_shift_out_of_range_traps_17002` uses an i32 result with `r=64`,
so the i64-width bound passes the test trivially — does not exercise the
[32, 63] gap.

Fix shape (NOT applied — read-only mode): derive `width = _INT_BITS.get(res.ty.name, 64)`
and use `r < 0 or r >= width`. Alternatively, mask `r` to `r & (width - 1)`
before the Python shift to match hardware semantics rather than trap. Either
preserves the cycle-21 propagation contract.

Confidence: 90. Reproducer confirmed end-to-end through `parse → lower → fold_module`.

---

## Rotation-area survey results

**helixc/frontend/parser.py** (1621 lines) — scanned for silent fallthroughs:
- `_eat` (line 91): raises ParseError on mismatch — loud.
- `_match` (line 96-99): returns None when token absent — by design, caller
  checks. Not silent failure.
- `_parse_int_literal` (line 375): `except ValueError → raise ParseError` — loud.
- Trait-method `continue`s (lines 226, 283, 292): correct end-of-clause flow,
  not silent skip.
- Postfix `continue` (line 1012): part of `while True` postfix loop, correct.
- Stdlib-merge `continue` (lines 1589, 1600): documented at the comment as
  "user takes precedence" / "missing-file warn-only when non-strict". Both
  print to stderr; only the second is silent-on-user-shadow (intentional).
- `_parse_string_attr_arg` (line 389): "skip any other tokens to RPAREN
  (lenient)" — documented lenient consumption. Intentional, not silent failure.

No new parser-side findings >= 75%.

**helixc/ir/passes/const_fold.py** (595 lines) — beyond C85-1 above:
- `except Exception: return None` at line 514 is gated by `except FoldError: raise`
  immediately above (cycle-21 C20-R1 fix) — confirmed not regressed.
- `FoldError`/`ShiftFoldError` body validation (cycle-30 C29-1/C29-2 + cycle-32
  C31-3): `raise TypeError` and `raise ValueError` are explicit, survive
  `python -O`. Confirmed not regressed.
- `_INT_BITS` table includes correct 32-bit aliases for isize/usize per
  cycle-20 C19-1 fix — confirmed not regressed.

No additional const_fold findings >= 75%.

---

## Verdict

**FAIL — 1 finding at conf >= 75% (C85-1, conf 90).**

Counter: cycle 85 (1 HIGH silent-failure) FAIL → reset to 0. Cycle 86 starts
toward 5-clean.

**No edits applied. This document is the sole write artifact.**
