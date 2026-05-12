# Audit Stage 28.9 cycle 92 — Type design

Scope: HEAD `d04e65b`.

Files read (read-only, no edits):

- `helixc/ir/tir.py` (full)
- `helixc/frontend/typecheck.py` (relevant ranges: 1097–1300, 1170–1290, 1820–1855, 2248–2285; full file is 43k tokens)
- `helixc/backend/x86_64.py` (relevant ranges: 215–270, 760–820, 897–940, 1144–1170, 1495–1600, 1900–2000)
- `helixc/ir/lower_ast.py` (cross-reference: 1030–1140, 1820–1830)
- `helixc/frontend/lexer.py` (cross-reference: 295–360)
- `helixc/frontend/parser.py` (cross-reference: 1085–1100)

Note: the audit prompt names `helixc/ir/builder.py`, which does not exist in
the repo at `d04e65b`. The IRBuilder + Op-construction surface lives in
`helixc/ir/tir.py` (class `IRBuilder` at lines 366–449; `Op` dataclass at
305–326; `OpKind` enum at 125–302). I audited that surface instead.

Prior-cycle deferred items NOT re-flagged:

- The general "no operand-result type invariant in `IRBuilder.emit()`"
  structural gap (cycle 78, "PASS with note") — pre-existing, scoped
  acceptance.
- Mixed-width binop / range bounds with i32 vs i64 sides reaching backend
  cmp / arith paths via the typecheck early-return for comparisons
  (`typecheck.py:1273-1274 → return TyPrim("bool")` without a
  same-type check), and the parallel laxness for `&&`/`||` lowering
  through MUL/ADD with bool result-type but iN operand types
  (cycle 76 F2/F3 at conf ~50/55; cycle 78 informational; cycle 80
  related). Out of scope for re-flag.
- TyMemTier strict-equality tier compare, lack of tier subsumption
  (cycle 5 F4 deferred).
- 3+-segment path → TyUnknown cascade carve-out (cycle 56 / 76 region).

## Finding F1 — IntLit accepts float-domain `type_suffix` and silently produces a float-typed integer-bit-pattern constant (conf ~85)

Reachable path:

1. **Lexer** (`lexer.py:328-354`) recognizes `_f16`/`_bf16`/`_f32`/`_f64`/
   `_fp8`/`_mxfp4`/`_nvfp4`/`_ternary` as valid suffixes on any numeric
   literal — including integer lexemes (those without `.` or `e/E`). The
   suffix whitelist (lines 338–341) is not domain-gated against
   `is_float`. So `42_f32` lexes as `Token(T.INT, int_value=42,
   type_suffix="f32")`.
2. **Parser** (`parser.py:1089-1091`) wraps the `T.INT` token as
   `ast.IntLit(value=42, type_suffix="f32")` — no domain validation.
3. **Typecheck** (`typecheck.py:1200-1202`):
   `return TyPrim(expr.type_suffix or "i32")` blindly trusts the suffix
   string. An `IntLit` carrying `type_suffix="f32"` is typed `TyPrim("f32")`.
   In `let x: f32 = 42_f32;`, `value_ty = declared = TyPrim("f32")`,
   `_compatible` accepts, and `_check_int_lit_fits` (1820–1845) returns
   early at line 1830–1831 because `"f32"` is not in `_INT_BOUNDS`. No
   diagnostic.
4. **Lowering** (`lower_ast.py:1037-1038`):
   `self.builder.const_int(expr.value, expr.type_suffix or "i32")` →
   `IRBuilder.const_int` in `tir.py:432-434` →
   `emit(OpKind.CONST_INT, result_ty=TIRScalar("f32"), attrs={"value":42})`.
   The IR now carries a `CONST_INT` op whose result is typed `f32`. No
   builder-side guard rejects this combination (CONST_INT with a
   non-integer scalar type).
5. **Backend** (`x86_64.py:1145-1154`): the CONST_INT emitter branches on
   `_is_i64_type(result.ty)`. `TIRScalar("f32")` is not i64, so it takes
   the else branch (line 1151-1153) and emits
   `mov eax, imm32` with `value & 0xFFFFFFFF` — i.e. the **integer**
   bit-pattern `0x0000002A` (for `42`) — into the slot. The result slot
   is tagged f32, so any downstream float arith / float cmp /
   `as f64` cast / load reads those 4 bytes as IEEE-754 f32 bits.
   `0x0000002A` as f32 ≈ 5.88e-44 (a denormal), not 42.0.

End-to-end effect: `let x: f32 = 42_f32; print(x);` (and analogous
`_f64`, `_bf16`, `_f16`) silently miscompile to a denormal/garbage
float value where the user clearly intended `42.0f32`. No diagnostic
fires at any layer.

Type-design root cause: the type-suffix domain is not validated against
the literal-kind anywhere in the pipeline. The lexer's flat whitelist
(`{i8..isize, u8..usize, bf16, f16, f32, f64, fp8, mxfp4, nvfp4,
ternary}`) treats all 18 names as legal regardless of whether the
preceding lexeme is `T.INT` or `T.FLOAT`. The corresponding symmetric
case — `FloatLit` with an integer suffix (`3.14_i32`) — has the same
shape: `typecheck.py:1204` returns `TyPrim(expr.type_suffix or "f32")`
trusting the suffix, then a float-valued AST node flows into a
construct expecting an integer; the float-to-int truncation is
implicit and surfaces (if at all) only as a `_compatible` mismatch
much later, with no domain-specific diagnostic.

Distinct from prior deferred mixed-width compare items (cycle 76 / 78
/ 80): those concern *same-domain* width mismatches (i32 ↔ i64) and
were marked sub-75. This finding is a *cross-domain* mismatch
(integer literal ↔ float type) that crosses CONST_INT/CONST_FLOAT
opcode boundaries — a structurally different defect, not covered by
prior cycles. The defect lives jointly in `lexer.py` (overbroad
suffix whitelist), `typecheck.py` (no kind-vs-suffix check on
`IntLit`/`FloatLit`), and `tir.py:IRBuilder.const_int` (no
dtype-vs-CONST_INT coherence check).

Suggested fix surface (not applied — read-only audit): either (a)
gate the lexer suffix-set by `is_float` so int lexemes only accept
integer suffixes and float lexemes only accept float suffixes; or
(b) reject the cross-domain combination in typecheck at
`_check_expr(IntLit)` / `_check_expr(FloatLit)` with a clear error
("integer literal cannot carry float suffix `_f32`; write `42.0_f32`
instead"). Backend-side and IR-side coherence checks would be
defense-in-depth but are not the natural fix point.

## Verdict

**FAIL** — 1 finding at conf ≥ 75% (F1 at ~85% confidence).

Stage 28.9 cycle 92 type-design audit: 0/1 PASS criterion not met.

STATE: NO EDITS. ONE Write only (this file).
