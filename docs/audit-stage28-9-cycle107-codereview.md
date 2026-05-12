# Audit Stage 28.9 cycle 107 — Code review

- **Date**: 2026-05-12
- **HEAD**: `6af8a46` ("Stage 28.9 cycle-106 fix-sweep: 4+ cycle-105 findings (full _is_i64_type sweep + cross-precision cast + unit + break/continue)")
- **Counter at start**: 0/5 (reset by cycle-106 fix-sweep after cycle-105 returned multiple HIGH findings).
- **Scope**: re-audit the cycle-106 fix-sweep diff (`git diff 77e4b85..6af8a46`):
  - `helixc/backend/x86_64.py:986` — param-spill prologue predicate flip `_is_i64_type` → `_is_64bit_int_type` (INT_SPILLS_64 path for u64/usize params).
  - `helixc/backend/x86_64.py:1198` (now 1216) — CONST_INT u64 path predicate flip, 8-byte `mov rax, imm64` for u64/usize constants.
  - `helixc/backend/x86_64.py:1234-1236` (now 1263-1265) — BITCAST `wide` classifier predicate flip on both source-type and dest-type halves.
  - `helixc/backend/x86_64.py:718-726` — new `cvtsd2ss_xmm0_xmm0` (`F2 0F 5A C0`) and `cvtss2sd_xmm0_xmm0` (`F3 0F 5A C0`) helpers.
  - `helixc/backend/x86_64.py:1332-1347` — new CAST arms: f64→f32 narrow via `cvtsd2ss`, f32→f64 widen via `cvtss2sd`.
  - `helixc/frontend/typecheck.py:336-351` — PRIMITIVES set drops `"()"` entry.
  - `helixc/frontend/typecheck.py:513-522` — `_resolve_type(TyName("()"))` short-circuits to `TyUnit()` before the PRIMITIVES set lookup.
  - `helixc/ir/lower_ast.py:1787-1797` — non-Range A.For arm now raises `NotImplementedError` instead of silently dropping into a single body lowering with the iter-var unbound.
  - `helixc/ir/lower_ast.py:1901-1918` — new explicit A.Break and A.Continue arms that raise `NotImplementedError`.
  - `helixc/tests/test_ir.py:282-401` — six new regression tests (C105 series): cvtsd2ss / cvtss2sd opcode discrimination, u64 CONST_INT 64-bit-path discrimination, loud-fail break, loud-fail continue, unit return-type compatibility.
- **Bar**: ZERO new findings at confidence < 80. CRITICAL or HIGH only.
- **Deferred-known list (carried forward, not re-flagged)**:
  - monomorphize `_mangle_ty` / hash_cons `_ast_equal` silent catchalls.
  - typecheck / struct_mono pre-flatten.
  - raw-200 enumeration in `parser.hx`.
  - `_is_i64_type` sibling deferred-known sites: BIT_AND / BIT_OR / BIT_XOR / SHL / SHR / BIT_NOT / NEG / SELECT / BR / RETURN / COND_BR / FFI_CALL plus the **cast-cascade matrix `from_is_i64`/`to_is_i64` @1282-1283** (predicates remain `_is_i64_type`-only post cycle-106).
  - `DT_BIND_NOW` unused constant.
  - Stage 29 K2 SIGILL probe scripts (`_probe_stage29_*.py` untracked).
  - `evaluator.hx` tag table (Stage-3 scope only).
  - All prior cycle findings.

---

## Methodology

Read-only inspection. No source mutation, no test runs, no scorecard. One Write of this doc.

Cycle-107 narrowly re-audits the cycle-106 fix-sweep commit `6af8a46` against the cycle-105 codereview / silent-failures / type-design findings. The three explicit user-supplied focus questions drive the analysis:

1. **BITCAST `wide` classifier symmetry** at lines 1263-1265: does the new predicate set correctly handle `bitcast<u64>(f64)` and `bitcast<f64>(u64)` symmetrically? And the usize-involved variants?
2. **CAST f32↔f64 register / arm placement**: do the new `cvtsd2ss` / `cvtss2sd` arms preserve xmm0 register state across the conversion, and is there a sibling CAST case (f64→i32, f32→i64) that the same emit-site convention got wrong?
3. **typecheck PRIMITIVES "()" drop blast radius**: does removing the textual `"()"` from the PRIMITIVES set break any previously-resolvable name path?

Concretely cross-referenced:

- `helixc/backend/x86_64.py:582-678` (movss / movsd load/store helpers used by the new CAST arms).
- `helixc/backend/x86_64.py:710-726` (cvttsd2si / cvtsi2sd / **new cvtsd2ss + cvtss2sd**).
- `helixc/backend/x86_64.py:983-1003` (FnCompiler.compile prologue — int param spill with the new `_is_64bit_int_type` gate).
- `helixc/backend/x86_64.py:1017-1055` (the `_is_*_type` predicates, including the cycle-102 `_is_64bit_int_type` helper).
- `helixc/backend/x86_64.py:1213-1251` (CONST_INT 8-byte path + CONST_FLOAT 8-byte split — unchanged).
- `helixc/backend/x86_64.py:1252-1272` (BITCAST `wide` classifier — the cycle-106 predicate flip).
- `helixc/backend/x86_64.py:1273-1359` (full CAST cascade — both old arms and the two new ones).
- `helixc/frontend/typecheck.py:95-103` (`TyUnit` frozen dataclass).
- `helixc/frontend/typecheck.py:336-351` (PRIMITIVES set with the "()" entry removed).
- `helixc/frontend/typecheck.py:513-522` (`_resolve_type` with the early `"()"` short-circuit).
- `helixc/ir/lower_ast.py:1787-1797` (non-Range A.For loud trap).
- `helixc/ir/lower_ast.py:1901-1918` (A.Break + A.Continue loud trap).
- `helixc/tests/test_ir.py:282-401` (the six new C105 regression tests).
- `helixc/bootstrap/parser.hx` + `helixc/bootstrap/kovc.hx` — confirmed via grep that the keywords `break` / `continue` and `for ... in` non-Range constructs do NOT appear as real syntax in the bootstrap sources (all matches are inside comments), so the new loud-trap arms cannot regress self-host compile.

---

## Findings table

| Severity   | Count |
|------------|-------|
| CRITICAL (90–100) | 0 |
| HIGH (80–89)      | 0 |
| MED (50–79)       | 0 reportable (below bar) |
| LOW (<50)         | 0 reportable (below bar) |

**Sub-threshold observations** (NOT findings — recorded for transparency only):

- **CAST cascade missing-arm for `f32 → i64` and `f64 → i64` widening conversions** (conf ~70, below threshold). The CAST cascade at `x86_64.py:1273-1359` still uses `_is_i64_type` (not `_is_64bit_int_type`) for the `from_is_i64` / `to_is_i64` predicates at lines 1282-1283. This was the cycle-105 carve-out item "cast matrix `from_is_i64`/`to_is_i64` @1253-1254 [_is_i64_type-only fallthrough sibling]" which the cycle-106 sweep deliberately did NOT touch — the sweep targeted CONST_INT, BITCAST `wide`, and param-spill only. However, a separate structural defect is now visible: even with `_is_i64_type` left alone (so `i64 IS in _is_i64_type`), the cascade has NO arm for `f32 → i64` or `f64 → i64`. Trace for `f32 → i64`: from_is_float=True, from_is_i64=False, to_is_i64=True. The cascade falls through to the `from_is_float and not to_is_float` arm at line 1327 (which emits `cvttss2si eax + mov_mem_rbp_eax` — 4 bytes only into an 8-byte i64 slot, leaving the high 4 bytes stale and NOT sign-extending). Same shape for `f64 → i64` at line 1321. This is structurally adjacent to the cycle-105 type-design F105-1 finding (cross-precision float casts) — the F105-1 fix landed two new arms for the f32↔f64 cross-precision case but did not address the `float → 64-bit-int` widening missing arms. The bug is pre-existing (predates cycle-106) and was not introduced or modified by the cycle-106 fix-sweep; the cycle-105 type-design audit did not enumerate it. Per cycle-107 scope rules ("audit cycle-106's broad fix-sweep"), this is out of scope to flag as a new cycle-107 finding — it is a latent cycle-105-type-design omission, not a cycle-106 regression. Recorded here below the 80-confidence bar; if the next type-design pass runs, this is a strong candidate for inclusion alongside the deferred `_is_i64_type` → `_is_64bit_int_type` flip of the cast-cascade predicates. (Confidence the bug exists: ~95. Confidence it should fire as a cycle-107 HIGH finding rather than a cycle-105 retroactive omission: ~50.)
- **CAST cascade `i64 → f32` / `u64 → f32` missing-arm** (conf ~70, below threshold). Symmetric to the above for the other direction. Trace for `i64 → f32`: from_is_i64=True, to_is_float=True, to_is_f64=False. Cascade: line 1285 (`from_is_i64 and not to_is_float and not to_is_i64`) → False (to_is_float=True). Line 1297 (`from_is_i64 and to_is_f64`) → False. Line 1304 (`from_is_i64 and to_is_i64`) → False. Line 1309 (`not from_is_float and to_is_f64`) → False (to_is_f64=False). Line 1315 (`not from_is_float and to_is_float`) → True (treats source as i32 — emits `cvtsi2ss xmm0, eax` after a 4-byte load, losing the high 32 bits of the i64). Same defect class as above; same disposition. Below 80.
- **Counter-reset semantics under prior FAIL** (conf ~90 that the disposition is correct, below 80 as a "finding"). The cycle-105 audits returned multiple HIGH findings across codereview / silent-failures / type-design; per the counter rules the audit-gate counter resets to 0/5. Cycle-106's fix-sweep landed the closures, and cycle-107 (this doc) is the first re-audit of the fixed surface. The counter should advance 0/5 → 1/5 on this cycle if CLEAN. No-finding observation.

---

## Cross-stage cuts examined

### BITCAST `wide` classifier symmetry (PASS at conf ≥ 90)

Pre-cycle-106 (line 1234-1236):
```
wide = self._is_f64_type(res_ty) or self._is_i64_type(res_ty) \
       or self._is_f64_type(op.operands[0].ty) \
       or self._is_i64_type(op.operands[0].ty)
```

Post-cycle-106 (line 1263-1265):
```
wide = self._is_f64_type(res_ty) or self._is_64bit_int_type(res_ty) \
       or self._is_f64_type(op.operands[0].ty) \
       or self._is_64bit_int_type(op.operands[0].ty)
```

Trace for all 8-byte BITCAST pairs:

| Cast | src_ty | res_ty | f64(res) | 64bit(res) | f64(src) | 64bit(src) | `wide` |
|------|--------|--------|----------|------------|----------|------------|--------|
| bitcast\<u64\>(f64)   | f64 | u64 | F | T | T | F | **T** ✓ |
| bitcast\<f64\>(u64)   | u64 | f64 | T | F | F | T | **T** ✓ |
| bitcast\<i64\>(f64)   | f64 | i64 | F | T | T | F | **T** ✓ |
| bitcast\<f64\>(i64)   | i64 | f64 | T | F | F | T | **T** ✓ |
| bitcast\<usize\>(f64) | f64 | usize | F | T | T | F | **T** ✓ |
| bitcast\<f64\>(usize) | usize | f64 | T | F | F | T | **T** ✓ |
| bitcast\<isize\>(f64) | f64 | isize | F | T | T | F | **T** ✓ |
| bitcast\<f64\>(isize) | isize | f64 | T | F | F | T | **T** ✓ |
| bitcast\<i64\>(u64)   | u64 | i64 | F | T | F | T | **T** ✓ |
| bitcast\<u64\>(i64)   | i64 | u64 | F | T | F | T | **T** ✓ |

All eight 8-byte permutations (and the i64↔u64 reinterpret pair) correctly take the `wide` path. The classifier is fully symmetric. PASS at conf ≥ 90.

The 4-byte BITCAST pairs (f32↔i32, f32↔u32, i32↔u32) all fail every predicate (none are f64 or 64-bit-int) and correctly take the 4-byte `mov_eax_mem_rbp` path. ✓

### New CAST arms (f32 ↔ f64) — register preservation + arm placement (PASS at conf ≥ 90)

**Opcode confirmation**:
- `cvtsd2ss xmm0, xmm0` per Intel SDM Vol. 2C: prefix `F2`, escape `0F`, opcode `5A`, ModRM `C0` (mod=11, reg=000=xmm0, r/m=000=xmm0) → `F2 0F 5A C0`. Matches Asm.cvtsd2ss_xmm0_xmm0 at line 721. ✓
- `cvtss2sd xmm0, xmm0` per Intel SDM Vol. 2A: prefix `F3`, escape `0F`, opcode `5A`, ModRM `C0` → `F3 0F 5A C0`. Matches Asm.cvtss2sd_xmm0_xmm0 at line 726. ✓

**Register preservation analysis**:
- f64 → f32 arm at lines 1338-1342: `movsd xmm0, [rbp+src]` → `cvtsd2ss xmm0, xmm0` (in-place, no other xmm register touched) → `movss [rbp+res], xmm0`. xmm0 contains the f32 result in its low 32 bits after the cvt; the movss store writes exactly those 32 bits to the result slot. ✓
- f32 → f64 arm at lines 1343-1346: `movss xmm0, [rbp+src]` → `cvtss2sd xmm0, xmm0` (in-place) → `movsd [rbp+res], xmm0`. xmm0 contains the f64 result in its low 64 bits; movsd stores 8 bytes. ✓

No xmm1 / xmm2-7 clobber. The in-place `xmm0, xmm0` form means register-pressure consideration is trivial — no scratch register needed. ✓

**Arm placement / interception analysis**:

The two new arms are inserted between the f32→i32 arm (line 1327) and the f64↔f64 same-precision copy arm (line 1349). Need to verify no earlier arm intercepts the new cases, and no later arm intercepts cases the new arms claim:

For `f64 → f32` (from_is_f64=T, to_is_f64=F, from_is_float=T, to_is_float=T):
- Line 1285 (i64→i32): from_is_i64=F → skip ✓
- Line 1290 (i32→i64): from_is_float=T fails `not from_is_float` → skip ✓
- Line 1297 (i64→f64): from_is_i64=F → skip ✓
- Line 1304 (i64→i64): from_is_i64=F → skip ✓
- Line 1309 (i32→f64): from_is_float=T fails `not from_is_float` → skip ✓
- Line 1315 (i32→f32): from_is_float=T fails `not from_is_float` → skip ✓
- Line 1321 (f64→i32): `not to_is_float` is False (to is f32, to_is_float=T) → skip ✓
- Line 1327 (f32→i32): `not to_is_float` is False → skip ✓
- Line 1338 (new f64→f32): T → **fires** ✓

For `f32 → f64` (from_is_f64=F, to_is_f64=T, from_is_float=T, to_is_float=T):
- Lines 1285, 1290, 1297, 1304: skip (i64-gated, from_is_i64=F)
- Line 1309 (i32→f64): `not from_is_float` is False → skip ✓
- Line 1315 (i32→f32): `not from_is_float` is False → skip ✓
- Line 1321 (f64→i32): from_is_f64=F → skip ✓
- Line 1327 (f32→i32): `not to_is_float` is False → skip ✓
- Line 1338 (new f64→f32): from_is_f64=F → skip ✓
- Line 1343 (new f32→f64): T → **fires** ✓

For `f64 → f64` (identity float copy, from_is_f64=T, to_is_f64=T):
- Line 1321: `not to_is_float` is False → skip
- Line 1327: `not to_is_float` is False → skip
- Line 1338 (new f64→f32): `not to_is_f64` is False (to_is_f64=T) → skip ✓
- Line 1343 (new f32→f64): `not from_is_f64` is False (from_is_f64=T) → skip ✓
- Line 1349 (f64→f64 8-byte copy): T → **fires** ✓

For `f32 → f32` (identity float copy, from_is_f64=F, to_is_f64=F):
- Line 1338: from_is_f64=F → skip
- Line 1343: to_is_f64=F → skip
- Line 1349: from_is_f64=F → skip
- Line 1355 (`from_is_float == to_is_float`): T → **fires** (4-byte mov_eax_mem_rbp / mov_mem_rbp_eax). ✓

The new arms slot in cleanly without intercepting any pre-existing path; identity-copy and same-precision-float-copy paths still reach their correct arms. PASS at conf ≥ 90.

### Param-spill prologue predicate flip (PASS at conf ≥ 90)

`x86_64.py:996-1003`:
```python
if self._is_64bit_int_type(p.ty):
    INT_SPILLS_64[int_idx](slot)
else:
    INT_SPILLS[int_idx](slot)
```

Pre-fix used `_is_i64_type`, which matched only `{i64, isize}`. For `u64` / `usize` params, the function entry would receive the full 64-bit value in `rdi` / `rsi` / `rdx` / `rcx` / `r8` / `r9` per SysV ABI, but the spill emitted `mov [rbp-slot], edi` (32-bit store), discarding the high 4 bytes. Post-fix the helper correctly routes all four 64-bit int names (`{i64, isize, u64, usize}`) to the 8-byte spill. ✓

INT_SPILLS_64 table cross-checked at `x86_64.py:980-990`:

<details>
<summary>SysV int param register spill table</summary>

| idx | INT_SPILLS_64 entry | x86 op |
|-----|---------------------|--------|
| 0   | `lambda d: asm.mov_mem_rbp_rdi(d)` | `mov [rbp+d], rdi` (rex.W 89 7D / 89 BD) |
| 1   | `lambda d: asm.mov_mem_rbp_rsi(d)` | `mov [rbp+d], rsi` (rex.W 89 75 / 89 B5) |
| 2   | `lambda d: asm.mov_mem_rbp_rdx(d)` | `mov [rbp+d], rdx` (rex.W 89 55 / 89 95) |
| 3   | `lambda d: asm.mov_mem_rbp_rcx(d)` | `mov [rbp+d], rcx` (rex.W 89 4D / 89 8D) |
| 4   | `lambda d: asm.mov_mem_rbp_r8(d)`  | `mov [rbp+d], r8`  (rex.WR 44 89 45 / 44 89 85) |
| 5   | `lambda d: asm.mov_mem_rbp_r9(d)`  | `mov [rbp+d], r9`  (rex.WR 44 89 4D / 44 89 8D) |

All six 64-bit spill targets emit rex.W (or rex.WR for r8/r9) prefixed `mov`, storing 8 bytes — matches SysV ABI for first six int params. ✓
</details>

### CONST_INT u64 path predicate flip (PASS at conf ≥ 90)

`x86_64.py:1213-1224`:
```python
if self._is_64bit_int_type(op.results[0].ty):
    self.asm.mov_rax_imm64(value)
    self.asm.mov_mem_rbp_rax(slot)
else:
    ...
```

Pre-fix: `u64`-typed `CONST_INT 12345_u64` would emit 32-bit `mov eax, 0x3039` into an 8-byte slot, leaving high 4 bytes stale (whatever was previously on the stack at slot+4..slot+7). Post-fix: 8-byte `mov rax, imm64` + 8-byte `mov [rbp+slot], rax` cleanly initializes the full 8-byte slot. ✓

The regression test `test_c105_u64_const_emits_64bit_path` at `test_ir.py:330-348` asserts the `48 B8` prefix (rex.W + `mov rax, imm64` opcode) appears in the emitted ELF. Discriminative — pre-fix the only `48 B8` source in a u64-only test would be absent. ✓

### Cross-precision float CAST arms (PASS at conf ≥ 90)

New arms at `x86_64.py:1332-1347` covered above. Regression tests:
- `test_c105_f64_to_f32_cast_emits_cvtsd2ss` at `test_ir.py:282-298` asserts `F2 0F 5A C0` byte sequence in compiled ELF.
- `test_c105_f32_to_f64_cast_emits_cvtss2sd` at `test_ir.py:301-313` asserts `F3 0F 5A C0` byte sequence.

Both byte sequences are 4 bytes and structurally unique within the compiled module — there's no other emit site for these exact prefixes. (The entry stub uses `mov eax, imm32; syscall`; main returns 0 via `mov eax, 0`.) Discriminative ✓.

### typecheck PRIMITIVES "()" drop (PASS at conf ≥ 95)

Pre-fix `PRIMITIVES = {..., "()"}` and `_resolve_type(TyName("()"))` reached the `if ty.name in PRIMITIVES: return TyPrim(ty.name)` branch, returning `TyPrim("()")`. Implicit-unit paths (function body that ends in `;` or has no trailing expr) produced `TyUnit()`. The dataclass __eq__ cascade between `TyPrim(name="()")` and `TyUnit()` is False (cross-class), so `_compatible` rejected a source-typed `fn foo() -> () { }` against its implicit-unit body even though both formally represent the unit type.

Post-fix: PRIMITIVES no longer contains `"()"`, and `_resolve_type` checks `if ty.name == "()": return TyUnit()` BEFORE the PRIMITIVES set lookup. Both source-typed and implicit-unit paths now converge on the singleton `TyUnit()`. Frozen dataclass `TyUnit` has no fields → `TyUnit() == TyUnit()` is always True under dataclass `__eq__`. ✓

**Blast-radius check** — anywhere else in the codebase that constructs `TyPrim("()")`:

```
$ grep -rn 'TyPrim("()") | TyPrim(.{0,3}\(\).{0,3})' helixc/
helixc/frontend/typecheck.py:345:    # TyPrim("()") in source-typed positions (e.g. `fn foo() -> () {}`)   (comment)
helixc/tests/test_ir.py:393:    TyPrim('()') while implicit-unit body produced TyUnit(), and the    (docstring)
```

No production-code site constructs `TyPrim("()")` — only the fix-comment and the test-doc reference. No downstream consumer was relying on the old representation. PASS at conf ≥ 95.

Regression test `test_c105_unit_return_type_compatible` at `test_ir.py:386-401` exercises `fn foo() -> () { }` end-to-end through parse + typecheck — pre-fix this raised `"type error: () does not match ()"`, post-fix it succeeds. ✓

### lower_ast non-Range For / Break / Continue loud-trap arms (PASS at conf ≥ 90)

**Bootstrap regression-risk check**:

```
$ grep -nE '^\s*(break|continue)\s*[;]?\s*$' helixc/bootstrap/*.hx
(no matches)
$ grep -nE '\b(break|continue)\b' helixc/bootstrap/*.hx
(all six matches are inside `//` comments)
$ grep -nE '^\s*for\s+\w+\s+in\s+' helixc/bootstrap/*.hx
(no matches)
```

Neither `break;` / `continue;` nor non-Range `for ... in <expr>` syntactic constructs appear in `parser.hx` or `kovc.hx`. The three new loud-trap arms cannot regress self-host compile, since the bootstrap doesn't exercise any of the now-trapped paths.

**Loud-trap correctness**:
- A.Break / A.Continue arms at `lower_ast.py:1901-1918` raise `NotImplementedError` with a span-anchored message, eliminating the prior catch-all-`return None` silent-drop at the bottom of `_lower_expr`. The cycle-105 silent-failures F1 finding documented that `loop { ...; if c { break; } }` pre-fix typechecked and emitted an infinite loop because Break fell through. ✓
- Non-Range A.For arm at `lower_ast.py:1787-1797` raises `NotImplementedError` with iter-expr-typename in the message. Pre-fix the dead code at this site lowered `iter_expr` once (discarding its return) then `body` once with the iter-var unbound. The post-fix loud trap converts silent miscompile to build failure. ✓

Regression tests `test_c105_break_in_loop_raises_loud` (test_ir.py:351-371) and `test_c105_continue_in_loop_raises_loud` (test_ir.py:374-384) exercise both arms; both assert NotImplementedError raised with the keyword name in the message. Discriminative ✓.

---

## Positive observations (no finding)

- **Cycle-106 fix-sweep precisely scoped to cycle-105 findings**. The six closures (CONST_INT u64, param-spill u64, BITCAST u64 wide, cross-precision float CAST, unit-type normalization, break/continue loud-trap, non-Range For loud-trap) map one-to-one onto the cycle-105 codereview / silent-failures / type-design findings. No scope creep, no opportunistic extra changes.
- **Defect class consistency** — the predicate flip `_is_i64_type` → `_is_64bit_int_type` is the same closure pattern cycle-100/102 applied to ADD/SUB/MUL. The cycle-102 helper `_is_64bit_int_type` (the disjunction `_is_i64_type ∨ _is_u64_type`) is the load-bearing abstraction; cycle-106 reuses it without copy-paste of the disjunction at the call sites. Maintenance-friendly: any future widening of `_is_u64_type` (e.g. to add a hypothetical `uptr`) automatically flows through.
- **Loud-trap-over-silent-miscompile** is the consistent fix shape for the silent-failures findings (Break / Continue / non-Range For). All three arms anchor the error to the source span (`expr.span.line:expr.span.col`) so the diagnostic is actionable rather than telemetry-only.
- **Six new regression tests** at `test_ir.py:282-401` provide byte-pattern discrimination for the new opcode emit (`F2 0F 5A C0`, `F3 0F 5A C0`, `48 B8`) and behavior discrimination for the loud-trap arms (NotImplementedError + keyword-in-message assertion). No test is a tautological pass.
- **Bootstrap non-regression** — none of the new loud-trap arms is reachable from `parser.hx` / `kovc.hx` syntax; heavy gate post-fix reports 1529 passed (+6 from new regressions), 0 actual failures.
- **No drift in the deferred-known set** — the cycle-107 carve-out list is exactly the cycle-105 + cycle-106 carry-forward set. No previously dispositioned item has been silently re-introduced.

---

## Verdict

**Verdict: CLEAN** — counter advances 0/5 → 1/5.

The cycle-106 fix-sweep cleanly closes the cycle-105 findings within the announced scope. The BITCAST `wide` classifier is symmetric across all 10 8-byte permutations (u64/i64/usize/isize × f64). The new f32↔f64 CAST arms place correctly in the cascade, use the right opcodes (`F2 0F 5A C0` / `F3 0F 5A C0` per Intel SDM Vol. 2), and operate in-place on xmm0 without disturbing any other SSE register. The PRIMITIVES "()" drop has no blast radius — no production-code site constructed `TyPrim("()")`. The Break / Continue / non-Range For loud-trap arms are unreachable from the bootstrap.

One sub-threshold observation (CAST cascade missing arms for `f32→i64` / `f64→i64` / `i64→f32` / `u64→f32` widening conversions) was identified but explicitly classified as a latent cycle-105-type-design omission rather than a cycle-106 regression — it predates cycle-106 and is structurally adjacent to (but not identical with) the deferred-known cast-cascade predicate-substitution carve-out. The bug exists at conf ≥ 95 but the "is this a cycle-107 finding rather than a cycle-105 retroactive omission" calibration is ~50; it sits below the 80-confidence audit-gate bar.

Re-flag inhibitor confirmed: no item in the cycle-107 scope is in the deferred-known set, and no previously dispositioned item has been re-introduced.

---

## Cross-reference

- **cycle 101** (`docs/audit-stage28-9-cycle101-codereview.md`): FAIL — F1 (missing regression test) + F2 (DIV/MOD/SHR signed-only).
- **cycle 102** (commit `26dfa82`): closed 3 of 4 cycle-101 findings.
- **cycle 103** (`docs/audit-stage28-9-cycle103-codereview.md`): CLEAN, counter 0/5 → 1/5.
- **cycle 104** (`docs/audit-stage28-9-cycle104-codereview.md`, HEAD `31e1725`): CLEAN, counter 1/5 → 2/5.
- **cycle 105** (`docs/audit-stage28-9-cycle105-codereview.md`, HEAD `77e4b85`): codereview CLEAN, counter 2/5 → 3/5. Silent-failures + type-design returned multiple HIGH findings → counter reset to 0/5.
- **cycle 106** (commit `6af8a46`): fix-sweep landed 6 closures across backend / typecheck / lower_ast. Heavy gate: 1529 passed.
- **cycle 107** (this doc, HEAD `6af8a46`): codereview **CLEAN**, counter 0/5 → 1/5.

---

## No code edits made

Strict read-only audit. One Write of this doc only. No Edit calls, no source mutation, no test run.
