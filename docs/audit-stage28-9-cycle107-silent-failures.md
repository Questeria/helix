# Audit Stage 28.9 cycle 107 — Silent failures

**Scope**: HEAD `6af8a46` (`Stage 28.9 cycle-106 fix-sweep: 4+ cycle-105
findings (full _is_i64_type sweep + cross-precision cast + unit +
break/continue)`).

**Mode**: STRICT READ-ONLY. No source edits performed. This document
is the sole write of the audit. Source files only read/grepped.

**Rotation (per prompt — avoid recently-audited surfaces):**

- Cycle-106 fix-sites verified for new silent failures introduced by
  the fix itself.
- SELECT / BR / RETURN / COND_BR / CALL u64/usize emit paths in
  `helixc/backend/x86_64.py` (the `_is_i64_type` siblings that the
  cycle-106 sweep did NOT update — explicitly in scope per the
  prompt's "SELECT/BR/RETURN/COND_BR/FFI_CALL u64 emit paths").
- LOAD_VAR / STORE_VAR / CAST u64 predicates.
- IR lowering bottom-of-`_lower_expr` `return None` for non-Break/
  Continue/StrLit/non-Range-For `A.Expr` subclasses (CharLit /
  StructLit-in-expr-pos / TileLit-in-expr-pos).
- typecheck `_resolve_type` "()" → TyUnit() normalization
  consistency.

**Deferred-known list (NOT re-flagged):**

monomorphize `_mangle_ty` / hash_cons `_ast_equal` silent catchalls;
typecheck/struct_mono pre-flatten in `check.py`;
`autotune.collect_autotuned_fns` missing `iter_fn_decls`;
`struct_mono.mangle_struct` collision;
`A.StrLit` IR-lowering gap (cycle-101 F1 deferred);
DIV/MOD signed-vs-unsigned emit (cycle-101 deferred);
SHR signed-vs-unsigned emit (cycle-101 deferred);
raw-200 enumeration in parser.hx Stage-8 monomorphize_pass
(cycle-7 deferred);
Stage 29 K2 SIGILL probes (untracked `_probe_stage29_*.py`);
DT_BIND_NOW unused constant in `elf_dyn.py`;
evaluator.hx tag table covers only Stage-3 evaluator subset;
`BIT_AND` / `BIT_OR` / `BIT_XOR` / `SHL` / `BIT_NOT` / `NEG` emit
paths in `helixc/backend/x86_64.py` (the `_is_i64_type` siblings
the cycle-106 sweep did NOT update — per prompt, deferred and not
re-flagged).

---

## Cycle-106 fix verification (regressions from the sweep itself)

Verified at HEAD `6af8a46`:

- `x86_64.py:1004` (param-spill `if self._is_64bit_int_type(p.ty)`),
  `x86_64.py:1222` (CONST_INT), `x86_64.py:1263-1265` (BITCAST
  `wide` classifier) all now route u64/usize through the 8-byte
  path. The three regression tests at `test_ir.py` (`test_c105_*`)
  inspect the byte patterns. PASS.
- `x86_64.py:1338-1347` (cvtsd2ss / cvtss2sd cross-precision CAST
  arms) emit before the `from_is_float == to_is_float` mov-copy
  fall-through. The narrow→wide / wide→narrow ordering is correct
  (the `not to_is_f64` / `not from_is_f64` guards prevent reaching
  these arms for same-precision f32→f32 / f64→f64 — which the f64
  arm at line 1349 handles with an 8-byte copy and the bottom
  4-byte arm at line 1355 handles for f32). PASS.
- `typecheck.py:519-520` (`_resolve_type` maps `A.TyName("()")` to
  `TyUnit()` before the PRIMITIVES check) and `typecheck.py:341-347`
  (`PRIMITIVES` drops `"()"`). Grep confirms no remaining
  `TyPrim('()')` construction in `helixc/` source (the only hit at
  `tests/test_ir.py:393` is inside a doc-comment string).
  PASS.
- `lower_ast.py:1901-1918` (A.Break / A.Continue raise
  NotImplementedError with line:col context) and
  `lower_ast.py:1788-1796` (non-Range For raises NotImplementedError)
  both depend on `expr.span.line` / `expr.span.col` — `A.Expr`'s
  base dataclass declares `span: Span` as a required field
  (`ast_nodes.py:99-101`), so the access is safe and cannot
  itself raise AttributeError on a well-formed AST. PASS.

The cycle-106 sweep itself introduces no new silent failures.

---

## Findings

### F1 — CALL int-arg u64/usize silently truncated by 32-bit reg-load (HIGH, conf 90)

**File**: `helixc/backend/x86_64.py`, CALL emit at lines 1809-1865;
the offending predicate at line 1848.

```python
if self._is_i64_type(arg.ty):
    INT_REGS_64[int_idx](arg_slot)
else:
    INT_REGS[int_idx](arg_slot)
```

`_is_i64_type` (line 1037-1043) is `{i64, isize}`. A `u64` /
`usize` typed arg therefore takes the else branch — `mov edi, [rbp
+ slot]` (a 32-bit move that zero-extends the destination
register-half-but-writes-only-32-bits-conceptually) instead of
`mov rdi, [rbp + slot]`. SysV ABI delivers the full 8-byte arg via
rdi/rsi/rdx/rcx/r8/r9; the 32-bit load discards the high 4 bytes of
the slot, so a u64 source value above 2^32 (or with any garbage in
the high half, since the high half is initialised by the CALLER's
upstream producer) reaches the callee truncated.

This is the exact same defect class as cycle-100 F1 / cycle-101 F2
/ cycle-105 F1-F3 — `_is_i64_type` name-equality missing the
pointer-width-alias siblings `u64`/`usize`. Cycle-106 swept
CONST_INT, param-spill, and BITCAST. The CALL int-arg arm and the
CALL return arm (F2 below) were not touched.

**Reachability**: `_lower_expr` at `lower_ast.py:1724` emits CALL
with `result_ty` taken from `callee_ir.return_ty` and arg Values
typed by the upstream producer. A user fn `fn id_u64(x: u64) -> u64
{ x }` followed by `let y = id_u64(0x1_0000_0001u64);` produces a
CALL op where `op.operands[0].ty == TIRScalar("u64")`. Pre-cycle-106
the literal already truncated at CONST_INT; post-cycle-106 the slot
holds the full 8 bytes — and the CALL arg-shuffle now silently
re-introduces the truncation that cycle-106's CONST_INT fix just
closed. The cycle-106 regression test
`test_c105_u64_const_emits_64bit_path` (`test_ir.py`) only inspects
CONST_INT bytes; no helixc-Python test executes a u64 CALL arg
through to exit-code verification.

**Sibling site (same root cause)**: the FFI_CALL arg arm at
`x86_64.py:1917` was explicitly fixed in cycle 77 (
`if self._is_i64_type(arg.ty) or self._is_u64_type(arg.ty):`) — so
extern "C" calls are correct. The internal CALL arm at line 1848
was the asymmetric sibling that escaped the cycle-77 sweep AND the
cycle-106 sweep.

**Recommended fix**: extend the predicate at line 1848 to
`self._is_64bit_int_type(arg.ty)` (or match the FFI_CALL pattern at
line 1917 textually). Regression test: caller passes
`0x1_0000_0001u64` in rdi, callee returns `a + 1u64`, harness asserts
result `== 0x1_0000_0002`.

### F2 — CALL return-value u64/usize silently truncated to eax (HIGH, conf 90)

**File**: `helixc/backend/x86_64.py`, CALL return-value path at
lines 1854-1864.

```python
if op.results:
    res_slot = self._slot_of(op.results[0])
    if self._is_f64_type(op.results[0].ty):
        self.asm.movsd_mem_rbp_xmm0(res_slot)
    elif self._is_float_type(op.results[0].ty):
        self.asm.movss_mem_rbp_xmm0(res_slot)
    elif self._is_i64_type(op.results[0].ty):
        self.asm.mov_mem_rbp_rax(res_slot)
    else:
        self.asm.mov_mem_rbp_eax(res_slot)
```

SysV returns u64 in the full 64-bit rax; this path reads only eax
(low 32 bits) when the result type is u64/usize, because
`_is_i64_type` excludes them. High 4 bytes of the 8-byte slot are
left uninitialised (`_alloc_slot` does not zero slots) or stale from
the previous use of that frame offset.

**Reachability**: any call whose registered fn return type is u64 /
usize. `fn id_u64(x: u64) -> u64 { x }` + `let y = id_u64(big);`
returns the value full in rax (callee's RETURN path is also broken
— see F3, but the callee saw and computed with the correct 8 bytes
because the ADD/SUB/MUL u64 paths are cycle-102-fixed). Caller reads
only low 4. Mirror of FFI_CALL return at line 1936 which IS correct
(`self._is_i64_type(op.results[0].ty) or self._is_u64_type(op.
results[0].ty)`).

**Recommended fix**: replace `self._is_i64_type(op.results[0].ty)`
with `self._is_64bit_int_type(op.results[0].ty)`, mirroring the
FFI_CALL return arm at line 1936.

### F3 — RETURN of u64/usize silently truncated to 32-bit load (HIGH, conf 90)

**File**: `helixc/backend/x86_64.py`, RETURN emit at lines 1954-1972;
the offending predicate at line 1962.

```python
if op.operands:
    slot = self._slot_of(op.operands[0])
    if self._is_f64_type(op.operands[0].ty):
        self.asm.movsd_xmm0_mem_rbp(slot)
    elif self._is_float_type(op.operands[0].ty):
        self.asm.movss_xmm0_mem_rbp(slot)
    elif self._is_i64_type(op.operands[0].ty):
        self.asm.mov_rax_mem_rbp(slot)
    else:
        self.asm.mov_eax_mem_rbp(slot)
```

Returning a u64-typed Value: `_is_i64_type` is false, so the load
is `mov eax, [rbp+slot]` — reads only the low 32 bits of the
result. Callee returns with only eax populated; the high 4 bytes
of rax are whatever was left over from the most recent op that
clobbered them (commonly 0, but not guaranteed — any preceding 64-
bit arithmetic in the same fn body leaves stale bits).

**Reachability**: `fn add_u64_const(a: u64) -> u64 { a + 1u64 }`.
Cycle-102 fixed ADD to use rex.W; cycle-106 fixed CONST_INT and
param-spill so `1u64` and the param both carry full 8 bytes. The
add result sits in an 8-byte slot. RETURN then 32-bit-loads it —
silent truncation of the very value cycle-102 + cycle-106 went to
lengths to compute correctly.

Sibling FFI_CALL return path is not affected (the FFI_CALL is the
caller, not the callee returning).

**Recommended fix**: replace `self._is_i64_type(op.operands[0].ty)`
with `self._is_64bit_int_type(op.operands[0].ty)`. Regression test:
`fn f() -> u64 { (1u64 << 40) | 7u64 }` linked into a main that
calls and exit-codes the low/high halves separately, harness asserts
both halves correct.

### F4 — SELECT u64/usize result silently truncated through eax (HIGH, conf 90)

**File**: `helixc/backend/x86_64.py`, SELECT emit at lines 1754-1808;
the offending predicate at line 1761.

```python
is_f64 = self._is_f64_type(res_ty)
is_i64 = self._is_i64_type(res_ty)
...
if is_f64: ... elif is_i64: self.asm.mov_rax_mem_rbp(a_slot)
else: self.asm.mov_eax_mem_rbp(a_slot)
...
if is_f64: ... elif is_i64: self.asm.mov_mem_rbp_rax(res_slot)
else: self.asm.mov_mem_rbp_eax(res_slot)
```

`is_i64` is `_is_i64_type(res_ty)` — false for u64/usize. The
load/store of both branches and the final store all run through
the 32-bit `eax` path. The high 4 bytes of `a` and `b` are dropped
and the high 4 bytes of `res_slot` are left stale.

**Reachability**: SELECT is emitted by IR builder for ternary-style
lowering (and by some passes — e.g. const-fold's select-of-const
rewrite). A program like `let x = if c { a_u64 } else { b_u64 };`
typically lowers to a BR-with-block-param merge (which routes to
the BR arm — see F5) rather than SELECT, but const-fold and other
passes can rewrite to SELECT post-lowering. Lower bound: any test
or program that exercises SELECT with a u64 result type.

The same predicate name-equality miss as F1-F3 / cycle-100-102-105.

**Recommended fix**: replace `is_i64 = self._is_i64_type(res_ty)`
with `is_i64 = self._is_64bit_int_type(res_ty)` (or rename the
local to `is_wide_int` to keep the meaning explicit).

### F5 — BR block-param u64/usize silently truncated (HIGH, conf 88)

**File**: `helixc/backend/x86_64.py`, BR emit at lines 1974-1997;
the offending predicate at line 1990.

```python
operand_ty = op.operands[0].ty
if self._is_f64_type(operand_ty):
    self.asm._movsd_load_xmmN(0, src_slot)
    self.asm.movsd_mem_rbp_xmm0(dst_slot)
elif self._is_i64_type(operand_ty):
    self.asm.mov_rax_mem_rbp(src_slot)
    self.asm.mov_mem_rbp_rax(dst_slot)
else:
    self.asm.mov_eax_mem_rbp(src_slot)
    self.asm.mov_mem_rbp_eax(dst_slot)
```

`A.If` lowering at `lower_ast.py:1726-1764` emits
`COND_BR → then_blk(value) → merge_blk(t_val)` and similarly for
`else_blk(e_val) → merge_blk(e_val)`. The merge block holds a
`new_block_param(t_val.ty, …)` — so if either arm computed a u64,
the merge block's param slot is u64-typed and the BR's operand is
u64-typed. The BR emit then uses `_is_i64_type` which is false for
u64, so the BR copy from `t_val`'s slot to the merge_blk's
param-slot is 32-bit, losing the high 4 bytes of the conditional
value.

**Reachability**: `let x = if c { a_u64 } else { b_u64 };` is the
exact lowering trigger — the merge block's u64 param receives
truncated values from both arms.

Same defect class. The COND_BR arm at lines 1998-2010 does NOT
copy values (only branches on the cond bool), so it's unaffected.

**Recommended fix**: replace the `_is_i64_type(operand_ty)` at
line 1990 with `_is_64bit_int_type(operand_ty)`.

### F6 — LOAD_VAR / STORE_VAR for u64/usize mutable locals silently truncate (HIGH, conf 90)

**File**: `helixc/backend/x86_64.py`, LOAD_VAR at lines 2016-2035,
STORE_VAR at lines 2036-2053; offending predicates at lines 2029
and 2047.

```python
# LOAD_VAR
elif self._is_i64_type(res_ty):
    self.asm.mov_rax_mem_rbp(var_slot)
    self.asm.mov_mem_rbp_rax(res_slot)
else:
    self.asm.mov_eax_mem_rbp(var_slot)
    self.asm.mov_mem_rbp_eax(res_slot)

# STORE_VAR
elif self._is_i64_type(src_ty):
    self.asm.mov_rax_mem_rbp(src_slot)
    self.asm.mov_mem_rbp_rax(var_slot)
else:
    self.asm.mov_eax_mem_rbp(src_slot)
    self.asm.mov_mem_rbp_eax(var_slot)
```

Mutable u64 locals: `let mut x: u64 = 1u64 << 40; x = x + 1u64;`.
The initial STORE_VAR (from the lowered CONST_INT + SHL — both
cycle-102-/cycle-106-correct in their own emit) writes only 32 bits
into the var slot. Subsequent LOAD_VAR reads only 32 bits. The
high-half of the var slot is whatever uninit/stale value it had.
Every read-modify-write cycle silently drops the high 32 bits.

The store-on-init slot itself is 8 bytes wide (`_alloc_slot` is
slot-uniform at 8 bytes for all types — line 897 area), so STORE_VAR
with a 4-byte mov leaves bytes 4..7 of the slot stale, NOT zero. A
later u64 ADD reading the slot via `mov rax, [rbp+slot]` (cycle-102-
fixed) loads the stale high half as part of its rax operand, then
`add rax, rcx` propagates the garbage.

This pattern is the same one cycle-105 F2 (param-spill) raised for
function entry; LOAD_VAR/STORE_VAR is the in-function-body sibling.

**Recommended fix**: replace both `self._is_i64_type(*)` predicates
at lines 2029 and 2047 with `self._is_64bit_int_type(*)`. Regression
test: `fn id_u64_via_mut(x: u64) -> u64 { let mut y = x; y }`
called with `0x1_0000_0001u64`, harness checks return value full 8
bytes.

### F7 — CAST arms involving u64/usize wrong (HIGH, conf 85)

**File**: `helixc/backend/x86_64.py`, CAST emit at lines 1273-1359;
the offending predicates at lines 1282-1283.

```python
from_is_i64 = self._is_i64_type(from_ty)
to_is_i64   = self._is_i64_type(to_ty)
```

The CAST dispatch table at lines 1284-1359 uses `from_is_i64` and
`to_is_i64` to select between the i64 / i32 / f64 / f32 arms. A
`cast<u64, i32>` (legal truncation) takes the bottom 4-byte mov-
copy at line 1355-1358 because none of the wide arms fire — the
low 32 bits do happen to be the correct truncation, so this case is
coincidentally non-corrupting. But:

- `cast<u32, u64>` (zero-extension widening): `from_is_i64` false,
  `to_is_i64` false. The arm at line 1290-1295 (`movsxd rax, eax`,
  intended for i32→i64) does NOT fire either — its guard `to_is_i64`
  is false. The cast falls to the bottom 4-byte mov-copy. High 4
  bytes of `res_slot` are left stale, NOT zero. A subsequent u64
  consumer reads garbage.
- `cast<i32, u64>` (sign-extension widening, debatable but
  consistent with i32→i64): `to_is_i64` false → bottom 4-byte mov.
  High 4 bytes stale.
- `cast<u64, f64>` / `cast<u64, f32>`: `from_is_i64` false → none of
  the i64→float arms (lines 1296-1301) fire. The cast falls to the
  i32→float arms (1308-1319), which load only 32 bits via
  `mov eax, [src]` and `cvtsi2sd xmm0, eax` — interpreting the low
  32 bits of the u64 as a SIGNED i32. Negative-looking high bits
  silently produce negative-signed float output for what semantically
  should be a positive u64.
- `cast<f64, u64>`: `to_is_i64` false → falls to f64→i32 arm at
  lines 1320-1325. Result is 32-bit; high 4 bytes of u64 dest slot
  stale.

Same `_is_i64_type` predicate name-equality class as F1-F6.

Note: a `cast<u32, u64>` correct semantics is zero-extension, not
sign-extension. The current `movsxd` at line 1293 is sign-extension
— wrong for the unsigned-narrow→unsigned-wide case even if the
predicate were extended. A complete fix needs a parallel unsigned-
widening arm (`mov eax, [src]` writes a 32-bit destination, which on
x86-64 implicitly zero-extends to rax; followed by
`mov [dst], rax`).

**Reachability**: any explicit `as` cast involving a u64/usize side.
The typecheck-hint at `typecheck.py:1799` ("use a bitcast through a
u64 intermediate") actively guides users into the broken CAST path
when the intent is a ptr<->int round trip.

**Recommended fix**: extend the predicates at lines 1282-1283 to
`_is_64bit_int_type`; add an unsigned-widening arm before the
`movsxd` arm at line 1290 that fires when the source type is
`_is_unsigned_int_type(from_ty)` and emits `mov eax, [src]` +
`mov [dst], rax`. Regression tests: `cast<u32, u64>`,
`cast<u64, i32>`, `cast<u64, f64>`.

### F8 — `_lower_expr` catch-all silently drops A.CharLit / A.StructLit (expr position) / A.TileLit (expr position) (HIGH, conf 82)

**File**: `helixc/ir/lower_ast.py`, bottom-of-`_lower_expr` at line
2235 (`return None`).

Cycle-105 fix added explicit `A.Break` / `A.Continue` arms and
flagged the non-Range `A.For` as the catch-all class of silent-
miscompile defects. The bottom `return None` at line 2235 STILL
silently swallows three other `A.Expr` subclasses that the parser
accepts and typecheck admits:

1. **`A.CharLit`**: typecheck at `typecheck.py:1268-1269` returns
   `TyPrim("char")`; lowering has no arm. `let c = 'A';` returns
   None from `_lower_expr`; `_lower_stmt`'s LetStmt path
   (`lower_ast.py:1031-1033`) catches None and substitutes
   `self.builder.const_int(0)`. The value of `'A'` (0x41) is lost.
   Downstream `c == 'A'` reads `0 == 0` → true for the WRONG
   reason (both sides folded to 0). Predicate-sensitive control
   flow flips silently.

2. **`A.StructLit` in expr position**: the LetStmt path at
   `lower_ast.py:848` has a special-case for `let x = S{a:1};` that
   bypasses `_lower_expr`. But a `StructLit` appearing as a Call
   arg (`f(S{a:1})`), an if-arm value (`if c { S{a:1} } else
   { S{a:2} }`), an Assign rhs (`y = S{a:1};`), or a Return value
   (`return S{a:1};`) routes through `_lower_expr` — bottom
   `return None`. The caller substitutes `const_int(0)`. The
   struct's field values are silently lost.

3. **`A.TileLit` in expr position**: same shape as StructLit. The
   let-stmt special-case at `lower_ast.py:762` handles
   `let t = tile<...>::...;`. A TileLit anywhere else in expression
   position silently lowers to `const_int(0)` via the same caller-
   side `or const_int(0)` substitution pattern that pervades
   `_lower_expr`.

These are the exact defect class the cycle-106 sweep set out to
close (`raise NotImplementedError` instead of silent fallthrough).
The cycle-106 fix added explicit arms for Break and Continue but
did NOT convert the bottom catch-all to a loud trap. A bottom-of-
`_lower_expr` arm of the form

```python
raise NotImplementedError(
    f"_lower_expr: no arm for {type(expr).__name__} "
    f"at {expr.span.line}:{expr.span.col}")
```

would catch StrLit (deferred-known), CharLit, StructLit, TileLit,
and any future-added `A.Expr` subclass uniformly — converting every
remaining silent fallthrough into a loud trap in one line. The
StrLit case is on the deferred-known list and the user has accepted
its silent miscompile; converting the catch-all to a loud trap
would close StrLit too as a side-effect, which is the desired
direction.

**Reachability**: any `char` literal in a let-stmt (very common in
helixc bootstrap source); any `S{...}` or `tile<>::...` in a
non-let-stmt position. CharLit is high-probability reachable from
helixc-Python tests that pattern-match on character values. StructLit-
in-expr-pos shows up wherever a struct literal is passed to a fn or
returned as an if-arm value.

**Recommended fix**: either (a) add explicit `A.CharLit` /
`A.StructLit` / `A.TileLit` arms with NotImplementedError, or
(b) replace the bottom `return None` at line 2235 with the
uniform `raise NotImplementedError` shown above. Option (b) is
the cycle-106-pattern generalisation.

---

## Summary

**8 findings at confidence ≥ 80%**, of two related root classes:

- **F1-F7** (seven HIGH findings at conf 85-90): the same
  `_is_i64_type` name-equality predicate miss for the u64/usize
  pointer-width-alias siblings that cycle-100, cycle-101, cycle-102,
  and cycle-105/106 have addressed in batches. Cycle-106 swept
  CONST_INT, param-spill, BITCAST. The CALL int-arg / CALL return
  / RETURN / SELECT / BR / LOAD_VAR / STORE_VAR / CAST arms remain
  on the unsafe predicate. Each is the same one-line `_is_i64_type
  → _is_64bit_int_type` swap; the CAST arm additionally needs an
  unsigned-widening arm for u32→u64.

- **F8** (one HIGH finding at conf 82): the bottom-of-`_lower_expr`
  `return None` at `lower_ast.py:2235` still silently drops at
  least three `A.Expr` subclasses (CharLit, StructLit-in-expr-pos,
  TileLit-in-expr-pos). The cycle-106 fix added explicit
  NotImplementedError arms for Break and Continue but did not
  convert the catch-all itself to a loud trap, leaving the same
  pattern reachable through other subclasses.

The cycle-106 byte-pattern regression tests
(`test_c105_f64_to_f32_cast_emits_cvtsd2ss`,
`test_c105_f32_to_f64_cast_emits_cvtss2sd`,
`test_c105_u64_const_emits_64bit_path`) pass at HEAD `6af8a46` —
they correctly test the byte-pattern emission of the fixed sites
but, as with the cycle-102 byte-pattern tests, they do not execute
the result. A u64-through-call-and-return runtime-execution test is
the missing coverage class.

---

## Findings table

| Severity | Count |
|---|---|
| CRITICAL (conf ≥ 90) | 0 |
| HIGH (conf ≥ 75)     | 8 (F1 90, F2 90, F3 90, F4 90, F5 88, F6 90, F7 85, F8 82) |
| MEDIUM (60-74)       | 0 |
| LOW (<60)            | 0 |

---

## Verdict

**Verdict: FINDINGS** — 8 HIGH/CRITICAL items, counter resets to 0/5.

---

## Cross-references

- Cycle-106 fix-sweep commit `6af8a46`: closed CONST_INT, param-
  spill, BITCAST for u64/usize (the cycle-105 silent-failures); +
  cvtsd2ss/cvtss2sd cross-precision float cast; + TyUnit
  normalization; + Break/Continue/non-Range-For loud traps. F1-F7
  of this cycle-107 audit are the cycle-106 SIBLING sites the
  sweep did not cover — same defect class. F8 is the
  catch-all-not-converted-to-loud-trap follow-up.
- Cycle-105 silent-failures `docs/audit-stage28-9-cycle105-silent-
  failures.md`: F1 (CONST_INT u64), F2 (param-spill u64), F3
  (BITCAST/CAST u64). Cycle-106 closed F1/F2 but only the wide-
  classifier portion of F3 — the CAST predicates (this audit's
  F7) were left on `_is_i64_type`.
- Cycle-102 fix-sweep `26dfa82`: closed ADD/SUB/MUL u64 only.
- Cycle-101 F2 / cycle-100 F1: the defect class root surfacing
  (cmp dispatch unsigned-vs-signed setcc / cmp dispatch i64-only
  64-bit path).
- Source-of-truth files: `helixc/backend/x86_64.py` predicates at
  lines 1037-1060 (`_is_i64_type` / `_is_u64_type` /
  `_is_64bit_int_type` / `_is_unsigned_int_type`); CALL 1809-1865
  (arg 1848, ret 1861); RETURN 1954-1972 (predicate 1962); SELECT
  1754-1808 (predicate 1761); BR 1974-1997 (predicate 1990);
  LOAD_VAR 2016-2035 (predicate 2029); STORE_VAR 2036-2053
  (predicate 2047); CAST 1273-1359 (predicates 1282-1283).
  `helixc/ir/lower_ast.py:2235` (bottom `return None`).
  `helixc/frontend/ast_nodes.py:307-313` (Break / Continue AST
  nodes); `helixc/frontend/typecheck.py:1268-1269` (CharLit
  typechecks).
