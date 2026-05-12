# Audit Stage 28.9 cycle 76 — Silent failures

**Scope.** Read-only HEAD `92ffc5a` (working at `b4e793c` which is the cycle-75 docstring polish; no source delta since cycle-74 `9a51cbf`). Narrow conservative scope. Prior C1–C75 + deferred-known not re-flagged.

**Criterion.** 0 findings at conf >=75%.

## Result: 1 finding at >=75% — FAIL

## Finding C76-1 — FFI_CALL float-argument silent miscompile

**Severity:** HIGH. **Confidence:** 80.

**Location.** `helixc/backend/x86_64.py` lines 1745–1756 (FnCompiler._emit_op FFI_CALL arm).

**Issue.** The argument-shuffle loop unconditionally routes every operand through `INT_REGS` (`rdi/rsi/rdx/rcx/r8/r9`) without checking `self._is_float_type(arg.ty)`. The asymmetric CALL arm at lines 1686–1697 does check and dispatches floats to `_movss_load_xmmN` / `_movsd_load_xmmN` per SysV. The FFI arm's omission silently miscompiles any `extern "C" fn f(x: f32)` invocation: `mov edi, [rbp+slot]` loads the 4 bytes of the f32 bit-pattern into `edi` and the callee receives garbage because SysV passes float args in `xmm0..xmm7`.

**Reachability probe** (run against HEAD):

```
extern "C" fn sinf(x: f32) -> f32;
fn main() -> i32 { let y: f32 = sinf(1.0_f32); 0 }
```

- `typecheck.typecheck(p)` → `[]` (clean).
- `lower_ast.lower(p)` produces `FFI_CALL target='sinf' operand_tys=[TIRScalar('f32')] ret_ty=TIRScalar('f32')`.
- `x86_64.compile_module_to_elf(mod, 'main')` → 5064-byte ELF, no error.

So the silent miscompile reaches the binary today. The return-type shape is partly addressed by the same arm's lines 1761–1766 (`# libc fns return int (eax) or pointer (rax). Stage 16.5 only wires int returns; other shapes deferred.`) — but even there it silently truncates float returns into `eax` rather than raising. The arg-side has no such deferral comment at all.

**Hidden errors.** Any FFI to math libc (`sinf`, `cosf`, `sqrtf`, `expf`, `logf`, `fabsf`, etc.) silently produces wrong values. No diagnostic, no trap-id, no test currently exercises a float-arg extern so the gap is invisible.

**Recommendation.** In the FFI_CALL arm, mirror the CALL arm's float-vs-int split:

```python
if self._is_float_type(arg.ty):
    if xmm_idx >= 8:
        raise NotImplementedError("FFI_CALL: >8 float args (Phase-0)")
    if self._is_f64_type(arg.ty):
        self.asm._movsd_load_xmmN(xmm_idx, arg_slot)
    else:
        self.asm._movss_load_xmmN(xmm_idx, arg_slot)
    xmm_idx += 1
else:
    # existing INT_REGS path
```

And mirror the float-return path on lines 1759–1766 using `self.asm.movss_mem_rbp_xmm0` / `_movsd_mem_rbp_xmmN`. If full SysV ABI lowering is out of scope for Phase-0, the minimum loud-fail fix is `raise NotImplementedError(f"FFI_CALL: float arg type {arg.ty} not supported (Phase-0); use a wrapper that takes/returns i32 bit-patterns)"` so the silent miscompile becomes an explicit compile error. Add a regression test in `tests/test_ffi.py` covering `extern "C" fn sinf(f32) -> f32`.

---

## Sub-threshold notes (conf <75)

- **N-1** (conf 70): the `_propagate_identities` transitive-closure loop at `const_fold.py:281–287` iterates `list(subst.items())` and mutates `subst[k] = subst[v.id]`. In SSA, cycles are unreachable (forward defs), so the inner while will terminate. Conf below threshold because no realistic input produces a cycle.
- **N-2** (conf 60, repeat of C65-N4): `x86_64.py:2833` `_emit_op` falls through with "Unsupported op — emit nothing (placeholder)" for unhandled `tir.OpKind`. Documented as Phase-0 stub for tensor / reduce / matmul ops that `lower_ast` does not currently emit. Re-flag would duplicate cycle-65 N-4.
- **N-3** (conf 55): `dce.py:101–105` seeds liveness only from `SIDE_EFFECT_KINDS` operands; a non-side-effect op whose result is read indirectly (e.g. through `STR_PTR.text` attr by a future pass) would be marked dead. No current pass introduces such indirection.
- **N-4** (conf 50): `fdce.py:30–31` silent no-op when `entry_fn` missing (repeat of C65-N4 fdce observation).
- **N-5** (conf 40): `cse.py:_op_hash` uses `repr(op.results[0].ty)` as the type key. Two distinct `TIRScalar('i32')` instances repr identically, which is the desired behavior; flagging only to document the assumption.

---

## Cycle-66..74 ASTVisitor migration stability — clean at >=75

The audit traced every `ASTVisitor` subclass and its instantiation site across `helixc/frontend/`:

- `deprecated_pass._DeprecationCallSiteCollector`: instantiated inside `find_deprecation_call_sites`, single use per call, no field state leaks across program-mutations.
- `grad_pass._GradCallFinder`: instantiated per `_expr_has_grad(e)` call, single-shot.
- `panic_pass._PanicCollector`, `_PanicArgsValidator`: instantiated locally inside `find_panic_call_sites` / `validate_panic_args` over a single read pass.
- `struct_mono._BodyVisitor`: instantiated ONCE inside `collect_concrete_uses` and reused across `prog.items` iteration via `visit_expr` shim. State on the visitor: NONE (the `seen` / `out` accumulators are closure-scope, not instance fields). Reuse is safe because `collect_concrete_uses` runs BEFORE struct_mono mutates the AST — no held-across-mutation hazard.
- `totality._SelfCallCollector`: instantiated per FnDecl scan; field `self.calls` is fresh per instance.
- `unsafe_pass._UnsafeBlockCollector`, `_RawPtrOpVisitor`: instantiated locally per `find_unsafe_blocks` / `find_raw_ptr_ops` call.

No site holds an ASTVisitor across a program-mutation event. The cycle-74 fix discipline (no explicit `generic_visit` in `visit_X` overrides; let the base class auto-descend) is consistently applied at every override I inspected (`_PanicCollector.visit_Call`, `_RawPtrOpVisitor.visit_UnsafeBlock` with push/pop context, `_BodyVisitor.visit_Let/visit_Cast/visit_Name/visit_TileLit/visit_ConstStmt`, `_SelfCallCollector.visit_Call`).

---

## cse/const_fold/dce/fdce — clean at >=75

- `cse.py`: PURE_KINDS set inspected; rewrite map block-scoped; `_op_hash` includes `result_ty_key + attrs_complex` (cycle-21 fix is intact). No new silent-failure pattern.
- `const_fold.py`: FoldError / ShiftFoldError prefix-discipline guarded by `__init__` (cycles 28/29/30/31/32 fixes intact). `_try_fold_op` arm-by-arm review: int / float / bitwise / shift / compare / NEG / BIT_NOT — every `except FoldError: raise` re-raise is present (cycles 21/28 fixes); generic `except Exception: return None` correctly bottoms out only after FoldError is re-raised.
- `dce.py`: SIDE_EFFECT_KINDS includes TRACE_ENTRY/EXIT (cycle 13), FFI_CALL (Stage 16.5 follow-up), TRAP, ARENA_PUSH/SET, TILE_INDEX_STORE — comprehensive.
- `fdce.py`: callee scan covers CALL/MODIFY/QUOTE.ast_pretty (free-ident regex). Roots include entry_fn + `is_pub` + `kernel` fns.

---

## Findings NOT re-flagged (deferred-known, per prompt scope)

- `monomorphize._mangle_ty` silent catchall — deferred.
- `hash_cons._ast_equal` SHA-256 fallback — deferred-known.
- `typecheck` pre-flatten in `check.py` — deferred.
- `struct_mono` pre-flatten in `check.py` — deferred.
- `autotune.collect_autotuned_fns` missing `iter_fn_decls` — deferred.

## Edits made

NONE. This audit was conducted in STRICT READ-ONLY mode. No source files were modified, no `Edit` calls were issued on any file. The only `Write` call was for this audit document at `docs/audit-stage28-9-cycle76-silent-failures.md`.

## Files inspected

- `helixc/ir/passes/cse.py` (full).
- `helixc/ir/passes/const_fold.py` (full).
- `helixc/ir/passes/dce.py` (full).
- `helixc/ir/passes/fdce.py` (full).
- `helixc/backend/x86_64.py` (op dispatch sweep at 906, 1144, 1664, 1724, 2547, 2580, 2619, 2792–2833).
- `helixc/frontend/ast_walker.py` (full).
- `helixc/frontend/{panic,unsafe,deprecated,grad,struct_mono,totality}_pass.py` — every `ASTVisitor` subclass and its instantiation site.
- `helixc/ir/lower_ast.py:1700–1725` (FFI_CALL emission).
- `helixc/frontend/typecheck.py:1119–1120` (extern body skip).

Reachability probe for C76-1 executed via `python -c` against HEAD with `helixc.frontend.parser.parse` + `helixc.frontend.typecheck.typecheck` + `helixc.ir.lower_ast.lower` + `helixc.backend.x86_64.compile_module_to_elf` — the float-arg extern compiled cleanly, confirming the silent miscompile.
