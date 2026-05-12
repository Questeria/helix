# Audit Stage 28.9 cycle 83 — Type design

Scope: HEAD `42f4e11` (Stage 28.9, 1/5 clean). Rotation-fresh files only:

- `helixc/ir/passes/cse.py` — hash key construction
- `helixc/frontend/diagnostics.py` — error rendering correctness
- `helixc/backend/ptx.py` — PTX type-to-machine mapping

Mode: STRICT READ-ONLY. Read/Grep/Glob/Bash only. No edits performed.
Prior C1–C82 findings and deferred-known items not re-flagged.

## Per-file review

### `helixc/ir/passes/cse.py`

Hash key `_op_hash` includes:
- `op.kind`
- `operand_ids` (post-rewrite, applied before key construction)
- primitive attrs (sorted)
- repr of non-primitive attrs (sorted) — covers CAST `from_ty`/`to_ty`
- `result_ty_key = repr(op.results[0].ty)` — covers bool-vs-i32 MUL discrimination (audit-10)

PURE_KINDS coverage vs. tir.OpKind: includes arithmetic (ADD/SUB/MUL/DIV/MOD/NEG),
bitwise (AND/OR/XOR/SHL/SHR/NOT), comparisons (CMP_*), CAST, BITCAST, MAXIMUM,
MINIMUM, POW, constants. Excludes side-effecting / nondeterministic / unspecified-
semantic ops (CALL, FFI_CALL, LOAD_*, STORE_*, ALLOC_*, BR/COND_BR/RETURN, ABS/EXP/
LOG/SQRT/RECIP per @safe deferred decision, TRAP, io.*, transform.*, agi.*).

DIV/MOD inclusion is sound: with identical operands the trap behavior is
identical, so eliminating the duplicate preserves observable semantics.

`seen[key] = list(op.results)` defensive copy (cycle-18 C18-C1 fix) is in
place. Block-scoped only — no cross-block dominance assumption.

Concern surveyed: an op with `results == []` would set `result_ty_key = None`
and could collide with another empty-results op of the same kind/operands.
All current PURE_KINDS produce at least one result, so unreachable in practice
— not a >=75% finding.

No conf>=75% findings.

### `helixc/frontend/diagnostics.py`

`render_caret` correctness:
- `use_color` honors NO_COLOR (per spec), HELIXC_COLOR override, then isatty()
- `_wrap` no-ops when `color=False` or empty styles; always pairs ANSI prefix
  with single reset
- `caret_pad = " " * max(0, col - 1)` — non-negative invariant on col≤0
- `caret = "^" * max(1, span_len)` — ensures at least one caret even on span=0
- Out-of-range `line` falls back to one-line message (`if not (1 <= line <= len(lines))`)
- `lines = source.splitlines()` and 1-indexed line lookup are consistent
- `did_you_mean` returns `[]` on empty name; otherwise delegates to difflib
- `Diagnostic.render` forwards all fields to `render_caret`
- `DiagSink.has_errors` checks for any level=="error" entry

Probed edge cases: col=0, col=-3, span_len=0 — all render sanely.

No conf>=75% findings.

### `helixc/backend/ptx.py`

Type-to-machine mapping (`_ptx_type_str`, `_DTYPE_SIZE`, `_DTYPE_PTX_LOAD`,
`_ld_reg_prefix`):
- `_ptx_type_str` mapping covers i{8,16,32,64}, u{8,16,32,64}, isize/usize→.b64
  (cycle-21 C20-1 fix), bool→.pred, f{16,32,64}, bf16. Default fallback `.b32`
  unreachable for current TIRScalar names.
- `_DTYPE_SIZE` includes isize/usize (cycle-21) and bool (cycle-35) — both
  match the canonical 64-bit-pointer-width and 1-byte-bool contracts established
  by typecheck.py.
- `_DTYPE_PTX_LOAD` likewise covers isize→s64, usize→u64, bool→u8.
- `_ld_reg_prefix`: f-family for f16/bf16/f32/f64; rd-family for i64/u64/isize/
  usize. i8/i16/i32/u8/u16/u32/bool → r (32-bit register). Bool→r is acceptable
  because bool is loaded as u8 and zero-extended into r32 by ptxas convention
  (no inferred miscompile path within Phase-0 scope).

Register-pool overflow guard (`_REG_POOL_CAP=256`) is enforced per-prefix in
`_new_reg`, matching the `.reg` declarations in `emit_kernel`.

Deferred-known surveyed and not re-flagged:
- `_format_param` treats all kernel params as `.b64` — documented v0.1 scope
- SCALAR_ADD/MUL/SUB/NEG dispatch on `%f`-prefix register to `.f32` regardless
  of underlying f64/f16 dtype — documented v0.1 "f32 only" scope and matches
  the SCALAR_CONST_FLOAT emitter
- SCALAR_CMP emits `setp.<cmp>.s32` only — comment at line 257 acknowledges
  float compares as future work

No fresh conf>=75% findings.

## Verdict

PASS — 0 findings at confidence >= 75%.

No edits performed (read-only audit).
