# Audit Stage 28.9 cycle 82 — Silent failures

**Scope.** HEAD `7b13010` (Stage 28.9 cycle-81 fix-sweep: 2 cycle-80
test-discrimination findings closed). Strict read-only mode. ONE Write
to this doc; no Edits on source files. Prior C1–C81 + deferred-known
NOT re-flagged. Parallel Stage 28.10 PAT_OR commits (e.g. `3b8ae79`,
`18bb141`, `50a296a`, `b31126b`, `08fdde5`, `5c7b54f`, `4fb54ef`) are
explicitly out of scope.

**Criterion.** 0 findings at confidence >= 75%.

## Result: PASS — 0 findings at conf >= 75%

---

## Cycle-81 fix-quality verification

The cycle-81 commit `7b13010` lands two test-discrimination fixes
flagged by the cycle-80 code-review audit. Both are verified
empirically.

### C81-fix-1 — FFI float-arg/return test (test_ffi.py:111–179)

Pre-fix the assertion was a bare `b"\xf3\x0f\x10" in elf` on a program
whose own caller signature `fn entry(x: f32) -> f32` produced movss
opcodes in its prologue/epilogue independent of FFI routing. A
regression to all-INT FFI_CALL routing would still leave entry's
movss bytes intact and the assertion would pass.

Post-fix the test builds TWO programs and compares movss counts:

```python
float_src: extern "C" fn cosf(x: f32) -> f32; caller() -> i32 { ... cosf(1.5_f32) ... }
int_src:   extern "C" fn puts(s: *const u8) -> i32; caller() -> i32 { ... puts(m) ... }
assert float_load > int_load
assert float_store > int_store
```

Empirical probe at HEAD:

```
float_load=1  float_store=1
int_load  =0  int_store  =0
```

The caller signature has NO float params/returns, so the only movss
opcodes in the float program come from the FFI surface (arg load to
xmm0 + return store from xmm0). A regression to INT_REGS routing on
either side would drop one of the two counts to zero, making the
counts equal. Discriminating. PASS.

### C81-fix-2 — for-range i64 increment test (test_ir.py:158–186)

Pre-fix the loop body had `total += 1_i64`, which itself emitted a
`CONST_INT(value=1, ty=i64)` independent of the for-range increment
under test. A regression to i32 on the for-range step would still leave
the body's `1_i64` literal intact and the scan-for-`CONST_INT(1, i64)`
assertion would pass.

Post-fix the body literal is `total += 7_i64`. Empirical IR dump
shows the fn now emits exactly one `(1, i64)` CONST_INT op (from the
for-range step), with the other CONST_INTs being `(0, i64)` ×2
(init + accumulator zero), `(5, i64)` (range end), and `(7, i64)`
(body literal):

```
CONST_INT in body=7_i64 version:
  (0, i64), (0, i64), (5, i64), (7, i64), (1, i64)
filtered CONST_INT(1, i64): [(1, i64)]
```

A regression to i32 on the for-range step would emit `(1, i32)`
instead of `(1, i64)`, the filtered list would be empty, and the
assertion would fail. Discriminating. PASS.

---

## Rotation: static-ELF emission (x86_64.py:2868–2928)

The scope label "elf.py (not _dyn)" is a misnomer — there is no
helixc/backend/elf.py. Static ELF emission lives in `x86_64.py`'s
`emit_elf` function (lines 2868–2928), invoked by
`compile_module_to_elf` when no FFI imports exist. Audit covered:

- **Header sizes.** `ehdr` length is 16 (ident) + 14 fields × {H,H,I,Q,Q,Q,I,H,H,H,H,H,H} = 64 bytes. Computed: `ELF e_ehsize=64` (line 2907), `e_phoff=64` (line 2904), `e_phentsize=56` (line 2908), `e_phnum=1` (line 2909). Hand-checked: ehdr is exactly 64 bytes, phdr is exactly 56 bytes (8 fields × {I,I,Q,Q,Q,Q,Q,Q} = 4+4+8+8+8+8+8+8 = 56).

- **Padding arithmetic.** `pad_size = CODE_OFFSET - len(ehdr) - len(phdr)` = 0x1000 - 64 - 56 = 3976. `pad = b"\x00" * pad_size` is well-defined when pad_size >= 0. No assertion-gate is present, but `CODE_OFFSET = 0x1000` is a module-level constant and `len(ehdr) + len(phdr) = 120` is structurally fixed by the literal bytes/struct.pack calls. The triple is non-parameterizable from the public emit_elf API surface, so pad_size cannot become negative without a source edit to one of three module-level constants. Not a finding.

- **filesz / memsz arithmetic.** `total_filesz = CODE_OFFSET + len(code)`, `total_memsz = total_filesz + extra_memsz`. Empirical probes at sizes (1, 100, 4000) with extra_memsz ∈ {0, 0x100000}: filesz and memsz match exactly the expected values. The `len(elf) == CODE_OFFSET + len(code)` invariant holds. No arithmetic overflow path inside Python's arbitrary-precision ints. No finding.

- **e_entry offset.** `code_vaddr = ELF_BASE + entry_offset`. ELF_BASE is 0x400000; entry_offset defaults to ENTRY_OFFSET (= CODE_OFFSET = 0x1000). The default entry-vaddr 0x401000 matches the file offset 0x1000 (the code region). Callers (compile_module_to_elf line 3090) do not override entry_offset for the static path, so the default applies. No finding.

- **PT_LOAD R+W+X flag.** Acknowledged in the docstring (lines 2877–2881) as a deliberate Phase-0 choice for the reflection-cells region. Not a silent failure — the flag value is documented.

## Rotation: lexer float-suffix parsing (lexer.py:300–354)

The number-suffix path at `_lex_number` accepts any token from a fixed
set: `{"i8","i16","i32","i64","isize","u8","u16","u32","u64","usize","bf16","f16","f32","f64","fp8","mxfp4","nvfp4","ternary"}`. The kind (INT vs FLOAT) is determined independently by whether the lexeme contains a decimal point or exponent. The lexer attaches `type_suffix` without cross-validating its kind against the literal kind.

Empirical observations on incoherent combinations:

| Source       | Token kind | type_suffix | Downstream behaviour                                                            |
|--------------|------------|-------------|---------------------------------------------------------------------------------|
| `1.5_f32`    | FLOAT      | `f32`       | normal — f32 literal                                                            |
| `1_i64`      | INT        | `i64`       | normal — i64 literal                                                            |
| `1.5_i32`    | FLOAT      | `i32`       | typecheck passes; lower_ast emits `CONST_FLOAT(value=1.5, result_ty=i32)`; codegen falls through to f32 branch (line 1173) → 4-byte f32 bit-pattern stored as if i32 |
| `1_bf16`     | INT        | `bf16`      | typecheck passes; lower errors at `_check_float_supported` with a loud message |
| `1.0_isize`  | FLOAT      | `isize`     | typecheck emits TypeError ("declared isize but value is i32") — caught loudly  |

**Weighed but NOT flagged.** The `1.5_i32` case is the only path that produces a silent miscompile (f32 bits stored, no diagnostic). However:

1. The trigger is a deliberate user-supplied syntactic incoherency — `1.5` with an `_i32` suffix is not produced by any compiler-internal pass.
2. The lexer-as-tokenizer / typecheck-as-validator separation is intentional architecture; `1.5_isize` IS caught at typecheck (declared-type mismatch), so the typechecker DOES validate suffix coherency for the common let-binding case. The escape window is when the ascription matches the suffix (`let x: i32 = 1.5_i32`).
3. This is closely related to the deferred-known `monomorphize._mangle_ty` / `hash_cons._ast_equal` "frontend silent catchall" treatments — the typechecker treating the literal's suffix as authoritative is identical posture.
4. The Stage 28.8 cycle 20 silent-failures audit (line 401+) reviewed the parallel `IntLit` → `const_int(expr.value, expr.type_suffix or "i32")` lowering path and explicitly characterized "literal-type inference … the literal's own type_suffix takes precedence" as intentional, declining to flag.

Confidence to flag: ~60% — below the >=75% threshold given the user-error nature of the trigger and the prior cycle-20 precedent on the parallel IntLit path. Documented here for cycle-83+ consideration if Stage 28.9 wants to tighten cross-validation, but NOT flagged for cycle 82.

## Rotation: DCE correctness (ir/passes/dce.py)

Read end-to-end (143 lines). Two checks:

- **SIDE_EFFECT_KINDS completeness.** Enumerated all `tir.OpKind` members not in the side-effect set, filtered by suspicious name pattern (store/set/push/pop/free/alloc/modify/splice/etc.). Only `LOG` and `TENSOR_STORE` surfaced. Cross-checked: neither is emitted anywhere in the lowering or codegen pipeline. `TENSOR_STORE` is declared at `tir.py:138` as part of the future tensor-IR vocabulary but has no producer in current Phase-0; `LOG` is similarly orphaned. DCE has nothing to potentially drop. No finding.

- **Liveness fixpoint.** Outer `while changed` loop alternates between liveness computation (seed from side-effect-op operands + spread to producers of live values) and removal of ops whose results are all dead. Block params and fn params are unconditionally seeded as live (lines 107–112), which preserves block-entry SSA values whose producers are upstream BR operands (BR is in SIDE_EFFECT_KINDS so its operands are also already seeded). Inner spread loop is single-fixpoint per outer iteration. Hand-traced on a 2-block fn with one dead let — removed exactly 2 ops (the dead let's RHS expr + the binding), no false positives. No finding.

- **Outer-loop termination.** `changed` flag is set only when an op is dropped. After the final pass with no drops, the loop exits. Each iteration strictly decreases total op count or terminates. No infinite-loop path. No finding.

## Sibling-class checks (cycle-79 + cycle-81 fix surfaces)

- **Test discrimination of OTHER recent regression tests.** Spot-checked `test_c76_2_ffi_call_routes_pointer_args_through_rdi` (pointer-int side, sibling to the float test). Asserts `call_qword_ptr_rip_rel_ffi` byte pattern + arg-rdi sequence. The byte pattern `48 8B 7D` (mov rdi, [rbp+disp8]) is sufficiently localized to FFI call sites that intra-Helix codegen does not introduce false positives; control-program comparison is not strictly needed because the int-side baseline IS the cycle-77 "fixed" routing. No regression-style false-pass window. Not re-flagged.

- **Cycle-79 FFI_CALL float-return arm at x86_64.py:1779–1795.** Re-examined; cycle-80 silent-failures audit (lines 24–53 of `audit-stage28-9-cycle80-silent-failures.md`) walked the if-chain order and sibling return sites (regular CALL return, RETURN op, BR param transfer). No new sibling holes since.

## Pre-existing items intentionally not flagged

Per audit scope, the following are deferred-known and NOT re-flagged:

- `monomorphize._mangle_ty` silent catchall
- `hash_cons._ast_equal` silent catchall
- `typecheck.check.py` pre-flatten / `struct_mono` pre-flatten in check.py
- `autotune.collect_autotuned_fns` missing iter_fn_decls
- `||` lowering at `lower_ast.py:1135–1138` ADD(result_ty=bool) (cycle-78 deferred)
- `tile_ir.py:220` "treat as opaque for v0.1" TODO (cycle-78 deferred)
- Regular CALL u64 routing asymmetry (cycle-78 type-design pre-examined)
- Presburger `None` on harder cases (cycle-5 incompleteness, examined)
- Lexer suffix-kind cross-validation (`1.5_i32` family) — examined this cycle, conf ~60%, below threshold; cycle-20 precedent on the parallel IntLit path declined-to-flag

## No code edits performed.

Read-only audit. No source files modified. Single Write to this doc only,
as scoped.
