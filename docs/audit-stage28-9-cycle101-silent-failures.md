# Audit Stage 28.9 cycle 101 — Silent failures

**Scope**: HEAD `fbfa211` (`Stage 28.13.1 cycle-2 fix: asymmetric probes for named-mode lookup`).

**Mode**: STRICT READ-ONLY. No source edits performed. This document is
the sole write of the audit. Source files only read/grepped.

## Cycle-100 fix verification

Verified in `helixc/backend/x86_64.py` post-`caf203f`:

- `_is_unsigned_int_type` predicate at line 1033 returns true for the
  full set `{u8, u16, u32, u64, usize}`. Membership matches the
  `_NUMERIC_INT_PRIMS` unsigned half in `typecheck.py:2138` and the
  `INT_LIKE_TYPES` row in `lower_ast.py:358`. PASS.
- Cmp dispatch at lines 1647-1666 picks the 64-bit path on
  `_is_i64_type OR _is_u64_type` on either operand, and chooses
  `unsigned_int_cmp_setters` (setb/setbe/seta/setae) whenever
  `_is_unsigned_int_type` matches either operand. PASS — cycle-100 F1
  and F2 both correctly remediated.

## Rotation targets

### dce.py (helixc/ir/passes/dce.py)

`SIDE_EFFECT_KINDS` preserves the full destructive/observable surface:
RETURN, BR, COND_BR, CALL, STORE_VAR, STORE_ELEM, ALLOC_VAR,
ALLOC_ARRAY, MODIFY, SPLICE, PRINT, QUOTE, REFLECT_HASH, ARENA_PUSH,
ARENA_SET, TILE_INDEX_STORE, FFI_CALL, TRAP, TRACE_ENTRY, TRACE_EXIT.
Liveness seeding from side-effect operands + fixpoint spread + result
predicate matches the documented algorithm in the module header.
Dead-store hazard would require a STORE_VAR or STORE_ELEM whose result
is unused — but neither op produces a result and both are in the
side-effect set, so they survive unconditionally. LOAD_VAR / LOAD_ELEM
are pure and correctly DCE-able only when their results are dead. No
finding.

### lower_ast.py StrLit lowering

(`helixc/frontend/strings_io.py` does not exist; rotation falls to
`lower_ast.py` per the prompt.) See F1 below.

### x86_64.py ADD/SUB/MUL integer-width handling

See F2 below.

## Findings

### F1 — Bare StrLit silently lowers to `const_int(0)` (HIGH conf 85)

**File**: `helixc/ir/lower_ast.py`, `_lower_expr` (lines 1036-2212),
and the `let`/`const`/enum-arg None-fallback sites at lines 815-816,
1012-1013, 1031-1032.

`typecheck.py:1253` accepts a bare `A.StrLit` as an expression and
returns `TyRef(TyPrim("char"), is_mut=False)` (an `&str`-ish type), so
`let s = "hi";` and `const NAME: ... = "x";` and the block-tail
position `{ ...; "literal" }` all pass typecheck. But `_lower_expr`
has no `isinstance(expr, A.StrLit)` arm — every existing reference to
`A.StrLit` in lower_ast (lines 1163, 1232, 1246, 1394, 1399, 1409,
1418, 1440, 1454, 1470, 1487) is a **call-site interception pattern**
matching `print_str(literal)` and friends, never a fallthrough path
for a bare literal. A bare StrLit reaching `_lower_expr` walks every
`isinstance` arm and falls through to the implicit `return None` at
the bottom of the function.

Each of `_lower_stmt`'s let/const arms then triggers the
`if v is None: v = self.builder.const_int(0)` defaulting (lines
1012-1013, 1031-1032). The binding `s` becomes a scalar i32 zero with
no diagnostic — same silent-corruption shape as cycle-100 F1, except
in the lowering layer not the backend. Calling `print_str(s)` afterward
would then fail to match the intercepted `Call(print_str, StrLit(...))`
shape (the arg is now a `Name`, not a `StrLit`), so the call lowers
silently as opaque-0 too. Net effect: typecheck-clean code with bare
StrLit at any non-intercepted expression position yields wrong runtime
behaviour with zero diagnostic.

Reachability: typecheck does not gate against bare StrLit in let/const
position; reproducer is `let s = "hi"; print_str(s);`. Confidence 85
because the typecheck branch and lower fallthrough are both
unambiguous; the only way this is not a bug is if Phase-0 deliberately
forbids the binding form, which the typecheck arm at 1253 contradicts.

### F2 — ADD/SUB/MUL backend dispatch ignores u64/usize, truncates to 32 bits (HIGH conf 92)

**File**: `helixc/backend/x86_64.py` lines 1318, 1343, 1368.

The arithmetic emit blocks for `ADD` (1304-1328), `SUB` (1329-1353),
and `MUL` (1354-1378) each dispatch on result type via
`self._is_f64_type → self._is_float_type → self._is_i64_type → else
(32-bit)`. `_is_i64_type` only matches `{i64, isize}` (per its
definition at 1025). u64/usize operands therefore fall through to the
32-bit `mov_eax_mem_rbp` / `add_eax_ecx` / `sub_eax_ecx` /
`imul_eax_ecx` path, silently truncating to 32 bits — **exact same
defect class** that cycle-100 F1 fixed in the cmp dispatch.

Reachability is concrete:

- `typecheck.py:336-343` includes u64/usize in `PRIMITIVES` and
  `typecheck.py:2138-2141` includes them in `_NUMERIC_INT_PRIMS`, so
  `let a: u64 = ...; let b: u64 = ...; a + b` typechecks.
- `lower_ast.py:1118-1119` emits `ADD/SUB/MUL/DIV/MOD` with
  `result_ty=l.ty`, propagating `u64` / `usize` to the IR op result.
- Backend dispatch sees `_is_i64_type(u64) == False` and falls through
  to the 32-bit path. The cycle-100 audit-trail commit message
  explicitly documents this defect class for cmp; the arithmetic
  triple has the same predicate-name-only check and has not been
  remediated.

The fix shape mirrors cycle-100: change each `elif self._is_i64_type
(op.results[0].ty):` to `elif self._is_i64_type(...) or
self._is_u64_type(...):`. (For MUL, `imul` is sign-agnostic for the
low 64 bits since 2's complement multiplication on equal widths
agrees in the truncated result, so a single 64-bit imul path is
correct. ADD/SUB are fully sign-agnostic on the truncated low word.)

Confidence 92 because the pattern, predicate, and type-system /
lowering coupling are identical to the cycle-100 F1 finding that just
passed audit-promotion.

## Verdict

**FAIL** — 2 findings at confidence ≥ 75%.

Findings count (conf ≥ 75): **2**.

- **F1** (HIGH conf 85): bare-`A.StrLit` falls through `_lower_expr`
  and silently becomes `const_int(0)` via the let/const/enum-arg
  None-fallbacks; typecheck accepts the binding form so user code is
  silently corrupted.
- **F2** (HIGH conf 92): ADD/SUB/MUL backend dispatch tests only
  `_is_i64_type`, so u64/usize arithmetic silently truncates to 32
  bits — same defect class as cycle-100 F1 (cmp), un-remediated in
  the arithmetic triple.

Stage 28.9 audit-gate counter remains pre-promotion; recommend a
cycle-102 fix-sweep on F2 (predicate widening, mechanical) and a
typecheck-or-lower disposition on F1 (either reject bare StrLit in
non-intercepted positions, or wire StrLit through CONST_STR / rodata
the way panic / print_str already do).
