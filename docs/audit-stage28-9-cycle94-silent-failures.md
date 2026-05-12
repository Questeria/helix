# Audit Stage 28.9 cycle 94 — Silent failures

**Scope:** HEAD `85bece0` (cycle-93 fix-sweep landing).

**Mode:** strict read-only — no source edits performed by this audit. Only
this single doc was written.

**Verdict:** **FAIL** — 2 findings at conf ≥ 75%.

---

## Cycle-93 fix verification

The cycle-93 fix adds literal-suffix kind-coherence checks in
`helixc/frontend/typecheck.py::_check_expr` for `IntLit` and `FloatLit`:

- `IntLit` with `type_suffix` in `_FLOAT_PRIM_NAMES = {"f16","bf16","f32","f64"}`
  → emits `TypeError_` ("integer literal has float-domain suffix").
- `FloatLit` with `type_suffix` in `_INT_PRIM_NAMES =
  {"i8","u8","i16","u16","i32","u32","i64","u64","isize","usize"}` →
  emits `TypeError_` ("float literal has integer-domain suffix").

Verified by direct grep and code reading. The hand-rolled `IntLit
'42'` + suffix `'f32'` case is reproducibly rejected by typecheck
post-fix. Both arms return a safe fallback type (`i32` / `f32`) so
downstream lowering still sees a kind-consistent type while the error
list is non-empty.

`_FLOAT_PRIM_NAMES ∪ _INT_PRIM_NAMES` covers exactly the set
`{i8,i16,i32,i64,isize,u8,u16,u32,u64,usize,f16,bf16,f32,f64}` — which
matches the lexer's recognised IEEE-style suffixes in
`helixc/frontend/lexer.py:338-340`. The fix is correct *for that
subset*. **However:** the lexer's suffix whitelist at
`lexer.py:341` also admits the quantized-float suffixes `fp8`,
`mxfp4`, `nvfp4`, plus `ternary` — these are absent from both
frozensets, which is the basis of F1 below.

---

## Findings

### F1 (HIGH, conf 90) — quantized-float suffixes (`fp8`, `mxfp4`, `nvfp4`) escape the cycle-93 kind-coherence check

Cycle-93 added `_FLOAT_PRIM_NAMES = {"f16","bf16","f32","f64"}` to
reject IntLit-with-float-domain-suffix at typecheck. The lexer (line
338-341 of `lexer.py`) *also* accepts the quantized-float suffixes
`fp8`, `mxfp4`, `nvfp4` (and the placeholder `ternary`), and
`typecheck.py:242-250` explicitly classifies `fp8`/`mxfp4`/`nvfp4` as
quantized **floats** (placed in the type lattice *above* every
integer, *below* `f16`/`bf16`).

A literal like `42_fp8` therefore:

1. lexes as `Token(INT, "42_fp8", int_value=42, type_suffix="fp8")`,
2. parses to `IntLit(value=42, type_suffix="fp8")`,
3. passes typecheck because `"fp8" not in _FLOAT_PRIM_NAMES` —
   `_check_expr` falls through to `return TyPrim("fp8")`,
4. lowers via `lower_ast.py:1037-1038` →
   `builder.const_int(42, "fp8")` (from `tir.py:432-434`) →
   `CONST_INT(result_ty=TIRScalar("fp8"))`,
5. x86_64 stores the raw int bit-pattern 0x2A into the fp8 slot.

This is **exactly the F1 silent-miscompile pattern** the cycle-93 fix
was designed to close — the fix is incomplete because it omits the
quantized-float domain. Empirically confirmed by running
`TypeChecker.check()` on `let x = 42_fp8;` and on `let x = 42_mxfp4;`
at HEAD `85bece0`: zero kind-coherence errors raised (only an
unrelated body-type-`()` error from the harness fn).

**Fix sketch:** extend `_FLOAT_PRIM_NAMES` to
`{"f16","bf16","f32","f64","fp8","mxfp4","nvfp4"}`. Symmetric question
for `ternary` (does the FloatLit→ternary path silently miscompile?) —
likely yes, but `ternary` semantics are less settled and a separate
decision.

### F2 (HIGH, conf 90) — `Parser._parse_autotune_int` silently drops digit-separator underscores

`helixc/frontend/parser.py:360-376` re-parses the integer literal from
the *raw lexeme* (`t.value`) by splitting on `_` and taking element
`[0]`, instead of using the already-parsed numeric `t.int_value`:

```python
s = t.value.split("_")[0]
...
return int(s)        # or int(s, 16) / int(s, 8) / int(s, 2)
```

Lexer-side, `lexer.py:303-326` accepts underscore-as-digit-separator
(`1_000_000` lexes with lexeme `"1_000_000"` and `int_value=1000000`).
The parser's split-on-`_`-take-`[0]` strategy was clearly written
assuming the only underscore in `t.value` is the suffix-introducer (so
`42_i32 → "42"`). For literals using digit separators, this silently
truncates to the first run of digits:

- `1_000_000` → `int("1")` = **1** (not 1_000_000).
- `0xFF_AA_i32` → `int("0xFF", 16)` = **255** (drops the AA byte).

Empirically confirmed at HEAD `85bece0`: parsing
`@autotune(block: [1_000_000, 64_i32]) fn f() {}` yields
`attrs=['autotune:block=1,64']` — the million silently became 1.
Severity: any autotune list using digit-separator underscores is
silently miscompiled before reaching `autotune.collect_autotuned_fns`,
which then picks a wildly wrong tile/block size.

**Fix sketch:** drop the split-and-reparse logic entirely — return
`t.int_value` directly (parser already does this for normal IntLits at
`parser.py:1091`). The only reason for the manual reparse appears to
be defensive vestige; `t.int_value` is computed by the lexer with
underscore-stripping already applied.

---

## Rotated targets — no findings

- **`helixc/ir/passes/dce.py` (liveness fixpoint termination):** the
  outer `while changed` loop only re-runs when at least one op was
  dropped this iteration, monotonically shrinking the op set; the
  inner `while spread` only adds to a bounded `live` set. Sound
  termination. The seed/spread/side-effect partition is consistent
  with the existing SIDE_EFFECT_KINDS allowlist (which has been
  expanded conservatively over many cycles).

- **`helixc/backend/elf_dyn.py` (PLT/GOT layout edge cases):**
  - SYSV hash bucket/chain construction at lines 257-275 is sound for
    `n_syms ≥ 1` (verified by hand for n_syms ∈ {1, 2, 3, ≥4}).
  - `.rela.plt` r_offset / r_info / r_addend triples correctly index
    GOT slots and dynsym entries (`sym_idx = i+1` skips the UND sym).
  - `.dynamic` entry count assertion (`n_dyn_entries == len(needed)
    + 12`) matches the literal-emitted entry count.
  - The no-imports edge case (which would produce malformed empty
    `.rela.plt` / `.got.plt` with non-null `.dynamic` entries) is
    unreachable: `x86_64.py:3066` gates `emit_elf_dyn` behind
    `buf.dyn.has_imports()`. Not a finding under Stage 28.9 scope.

- **`helixc/frontend/parser.py` rare-path rules:** the F2 finding
  above came from here. Other `try/except` blocks (lines 367-376
  re-raise as ParseError; 707/736 use the `_no_cmp_lt_gt` counter
  with proper inc/dec via `try/finally`-equivalent flow; 1306-1330
  speculative-parse with `save_i` restore) are all sound.

---

## Verdict

**FAIL** — 2 findings at conf ≥ 75% (F1 quantized-float suffix
escape, F2 autotune-int digit-separator silent drop).

## Notes

- This audit performed no source edits. The only Write call produced
  this document.
- Prior-cycle deferred-known issues (`monomorphize._mangle_ty` /
  `hash_cons._ast_equal` catchalls, `typecheck/struct_mono`
  pre-flatten, `autotune.collect_autotuned_fns` missing
  `iter_fn_decls`, `struct_mono.mangle_struct` collision) were not
  re-flagged per scope.
- Parallel "Stage 28.10/28.11" commits are independent and out of
  scope.
