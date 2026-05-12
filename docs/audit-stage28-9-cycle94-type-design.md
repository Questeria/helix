# Audit Stage 28.9 cycle 94 — Type design

Scope: HEAD `85bece0` (Stage 28.9 cycle-93 fix-sweep). Strict read-only.
Verify cycle-93 type-surface (F1 IntLit/FloatLit kind-coherence) +
rotate fresh: grad_pass walker totality, lower_ast Cast result-type,
x86_64 register/slot type-class invariants. Cycles C1–C93 + deferred-
known omitted.

Verdict: **FAIL** — 1 finding at confidence ≥ 75%.

## F1 — `_FLOAT_PRIM_NAMES` / `_INT_PRIM_NAMES` omit quantized float family (HIGH, conf 90)

`helixc/frontend/typecheck.py:379-383` defines:

```python
_FLOAT_PRIM_NAMES = frozenset({"f16", "bf16", "f32", "f64"})
_INT_PRIM_NAMES   = frozenset({"i8","u8","i16","u16","i32","u32","i64","u64",
                               "isize","usize"})
```

The lexer (`helixc/frontend/lexer.py:338-341`) accepts a wider set of
numeric type suffixes:

```python
{"i8","i16","i32","i64","isize",
 "u8","u16","u32","u64","usize",
 "bf16","f16","f32","f64",
 "fp8","mxfp4","nvfp4","ternary"}
```

The four suffixes `fp8`, `mxfp4`, `nvfp4`, `ternary` are valid lexer-
recognised numeric-literal suffixes but appear in **neither** the
float set nor the integer set used by the cycle-93 kind-coherence
checks at `_check_expr` (typecheck.py:1220, 1233).

The typecheck rank table (typecheck.py:242-253) and the comment block
above it explicitly classify `fp8` (rank 45), `mxfp4`/`nvfp4` (rank
43) as **quantized floats** — they "live ABOVE every integer" so
`D<fp8> + D<i64>` widens to `fp8`. They are members of `PRIMITIVES`
at line 341 alongside `f16/bf16/f32/f64`.

Consequence: `42_fp8` (an `A.IntLit` with `type_suffix="fp8"`) still
slips past the cycle-93 check, returns `TyPrim("fp8")` from
`_check_expr`, lowers via `lower_ast.py:1038`
`builder.const_int(42, "fp8")` to `CONST_INT(result_ty=
TIRScalar("fp8"))`, and the x86_64 backend at `x86_64.py:1145-1153`
takes the `else` branch (not i64/isize) and stores the raw 32-bit
integer bit-pattern `0x2A` into the fp8 slot — the exact cross-
domain miscompile shape that F1 was introduced to prevent, just
for the quantized-float family rather than `f32`. Symmetric concern
for `1.5_isize`-style FloatLit + integer-domain suffix is already
covered, but FloatLit with `fp8`/`mxfp4`/`nvfp4` suffix would also
be malformed (a quantized representation cannot generally hold an
arbitrary FloatLit value losslessly, and the IR `const_float`
backend path at `x86_64.py:1161-1177` only knows f32/f64 layouts,
so a quantized FloatLit silently bit-packs as IEEE-754 f32). The
`ternary` suffix is undecided in the rank table and has no clear
domain assignment in the type-coherence layer.

Suggested fix shape: extend `_FLOAT_PRIM_NAMES` to
`{"f16","bf16","f32","f64","fp8","mxfp4","nvfp4"}` and add an
explicit decision for `ternary` (either include in `_INT_PRIM_NAMES`
or reject every literal suffix on numeric literals until the type
is wired through). Add regression tests
`test_c94_intlit_with_fp8_suffix_rejected`,
`...mxfp4_suffix_rejected`, `...nvfp4_suffix_rejected`.

Confidence: 90. The lexer/typecheck/lower/backend path traces
end-to-end as the F1 case; the only difference is which suffix
name reaches the check.

## Rotated fresh — no findings ≥ 75%

- **grad_pass.py dispatch totality**: the file's own docstring
  (lines 30–46) explicitly defers walker-drift in
  `_rewrite_in_expr` / `_resolve_in_expr` as a known design
  trade-off (returning new nodes / mutating in place, which
  ASTVisitor's read-only contract cannot express). Per scope
  rules, deferred-known is not re-flagged.
- **lower_ast.py Cast derivation** (lines 2135–2143): result_ty
  is the lowered syntactic target_ty; from_ty is the lowered
  source `inner.ty`. No derivation step that could mis-classify.
- **x86_64.py register-allocator type-class invariants**:
  the backend has no register allocator — all values live in
  stack slots. The only type-class distinction is i64/isize
  via `_is_i64_type` (already widened cycle-19 C18-1) and f64
  via `_is_f64_type`. No invariant violation observed.

## Counter

Cycle 94 finds 1 HIGH finding. Counter → reset.

**No edits performed by this audit.** One Write (this doc) only.
