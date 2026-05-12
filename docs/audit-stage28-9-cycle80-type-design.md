# Audit Stage 28.9 cycle 80 — Type design

**Scope.** HEAD `d218e65` (Stage 28.9 cycle-79 fix-sweep: FFI float-return via
xmm0 + 2 regression tests). Strict read-only mode. ONE Write to this file; no
Edits. Prior C1–C79 findings + deferred-known set NOT re-flagged.

**Criterion.** 0 findings at conf >= 75%.

## Result: PASS — 0 findings at conf >= 75%

## Verification of cycle-79 fix surface

### FFI_CALL return-handler conditional order (x86_64.py:1779–1794)

The post-fix chain at the FFI return site is:

```python
if   self._is_f64_type(op.results[0].ty):     movsd_mem_rbp_xmm0(res_slot)
elif self._is_float_type(op.results[0].ty):   movss_mem_rbp_xmm0(res_slot)
elif self._is_i64_type(...) or self._is_u64_type(...): mov_mem_rbp_rax(res_slot)
else:                                          mov_mem_rbp_eax(res_slot)
```

Per the type-predicate helpers at lines 999–1017:

- `_is_f64_type` = name == `"f64"`
- `_is_float_type` = name in `("f16","bf16","f32","f64")`
- `_is_i64_type` = name in `("i64","isize")`
- `_is_u64_type` = name in `("u64","usize")`

**Mutual exclusion.** The four name-sets are disjoint. The branch-order
short-circuit makes the f64-first / f32-second pair effectively read as

- arm 1: `name == "f64"`
- arm 2: `name in ("f16","bf16","f32")` (since f64 already taken)
- arm 3: `name in ("i64","isize","u64","usize")`
- arm 4: anything else (i32/u32/i16/u16/i8/u8/bool/pointer-shaped-not-u64).

f16/bf16 entering arm 2 is structurally moot because
`_check_float_supported` (x86_64.py:1019) errors on f16/bf16 well before
FFI_CALL codegen runs — the FFI return path is unreachable with those
types.

**Totality.** Any scalar type is matched by at most one arm and the `else`
covers everything not in the prior three. Total over `TIRScalar`. Aggregate
types (struct return, tuple return) bypass this site entirely (FFI_CALL in
Phase-0 only ships int/ptr-shape and float-scalar returns; struct-return ABI
is a separate lowering path not exercised here). No partition gap at the
cycle-79 surface.

### Test byte signatures `F3 0F 10` / `F3 0F 11` (test_ffi.py:140,145)

These are the Intel SDM Vol 2B encodings for `MOVSS xmm, m32` (load,
opcode `F3 0F 10 /r`) and `MOVSS m32, xmm` (store, `F3 0F 11 /r`), which
the SysV AMD64 psABI mandates for moving a `float`-class scalar into / out
of xmm0 across an FFI boundary. They are not coincidental:

- `Grep "0xF3, 0x0F, 0x10|0xF3, 0x0F, 0x11"` across `helixc/`: matches only
  `helixc/backend/x86_64.py` (four helpers — `movss_xmm0_mem_rbp`,
  `movss_xmm1_mem_rbp`, `movss_mem_rbp_xmm0`, plus the generic
  `_movss_load_xmmN` / `_movss_store_xmmN` for SysV arg passing). No other
  instruction emitter in the backend can produce these byte patterns.

- Empirical: a control ELF compiled from the int-only `puts(*const u8) -> i32`
  FFI hero test (no f32 anywhere) does NOT contain either of the byte
  patterns in its full 5064-byte payload (ELF header, four phdrs,
  .text, .got/.got.plt, .dynsym/.dynstr/.rela.dyn/.dynamic, string
  literals). The opcode prefix bytes are not in puts's hello-world idiom
  and not in the ELF metadata for a Phase-0 FFI binary.

A regression on the cycle-77 arg-side that re-routed f32 through INT_REGS
would lose the F3 0F 10 load opcode at the call site; a regression on
the cycle-79 return-side that reverted to `mov eax -> slot` would lose
F3 0F 11. The bare `in elf` assertion is therefore discriminative for
both ABI surfaces. PASS.

## Rotation: lower_ast.py range expression lowering

Cycle-77 fixed the for-range CONST_INT(1) dtype to match the iterator type
(lower_ast.py:1845–1854). Rotate fresh to other range edge cases.

- **Inclusive `..=` in expression context** — `parser.py:_parse_range`
  (line 970) only matches `T.DOTDOT`, never `T.DOTDOTEQ`. A program
  `for i in 0..=5 { ... }` errors at parse time with
  `expected LBRACE got DOTDOTEQ '..='`. The `DOTDOTEQ` token is consumed
  only in pattern contexts (parser.py:1400, 1453 → `PatRange`).
  Behavior: fail-loud, not a silent type-design defect.

- **Stride ranges** (`0..10 step 2`, `(0..10).step_by(2)`, etc.) — not
  in the grammar; not parsed. AST has no stride field on `Range`. No
  silent surface.

- **Reverse ranges** (`5..0`, descending) — parser accepts; IR lowers
  to the same scheme as forward ranges; the loop header
  `CMP_LT(iter, end)` fires false immediately, body never runs (zero
  iterations). This matches Rust's exclusive-range semantics for
  `5..0`. No silent type confusion; user wanting descending iteration
  uses an explicit `while`.

- **Mixed-width range bounds** (`for i in 0_i64..n_i32 { ... }`) —
  deferred-known per `audit-stage28-9-cycle76-type-design.md` line 146
  (conf ~50). NOT re-flagged.

- **Empty Range expr** (`x..`, `..y`) — `Range` AST has
  `Optional[start]` / `Optional[end]`. The for-loop lowering at
  lower_ast.py:1787 explicitly rejects them via the `start is None or
  end is None` guard and falls through to the no-op path. No silent
  miscompile; explicit deferral. No fresh finding.

## Rotation: typecheck.py internal correctness (narrow)

- `Range` (typecheck.py:1645–1650) checks `start` and `end`
  independently and returns `TyUnknown(hint="range")`. No
  same-type / width / signed-class constraint between bounds.

- `For` (typecheck.py:1627–1637) binds the loop variable to
  `iter_ty` directly (the iterator expression's type), which for a
  Range is `TyUnknown`. So the loop variable enters the body scope
  with `TyUnknown` and downstream uses fall through the
  best-effort path. This is permissive-by-design — the IR lowering
  has authoritative dtype information from the bounds' inferred types
  (`start_v.ty` at lower_ast.py:1809), and the cycle-77 fix already
  makes the increment dtype-consistent. No fresh ≥75% type-design
  defect on the typecheck side.

## No edits

This audit performed read-only inspection via Read, Grep, Glob, and Bash
(parser smoke-probe for `..=`, mixed-width range IR dump, and ELF byte
collision check). The single Write was this audit doc. No source files
in `helixc/` were modified.

## Heavy gate note

Heavy gate at HEAD `d218e65` (per commit message): 1509 passed, 0
failures. Cycle 80 starts toward 5-clean.
