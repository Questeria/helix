# Helix Compiler — Code Quality Audit: Stages 9–16.5

**Repo:** `C:/Projects/Kovostov-Native` at commit `3421b21` (original audit)
**Date**: 2026-05-08 (original); recreated 2026-05-10 from subagent task notification
**Scope:** Stages 9 (closures), 10 (modules), 11 (reflection), 12–14 (autodiff), 14.5 (@checkpoint), 15 (tile/tensor), 16 (PTX), 16.5 (FFI)
**Files reviewed:** parser.hx, kovc.hx, frontend/parser.py, flatten_modules.py, lower_ast.py, tir.py, passes/dce.py, passes/fdce.py, backend/ptx.py, backend/elf_dyn.py, backend/x86_64.py, autodiff.py, autodiff_reverse.py, grad_pass.py

---

## Summary

- **11 confirmed issues** across all 8 stage groups
- **4 CRITICAL (stop-the-line)**
- **4 MEDIUM**
- **3 LOW**

## Resolution Status (updated 2026-05-10)

| # | Severity | Finding | Status | Commit |
|---|----------|---------|--------|--------|
| CRITICAL-1 | CRITICAL | FFI_CALL missing from DCE SIDE_EFFECT_KINDS | ✅ FIXED | 6bda4b3 |
| CRITICAL-2 | CRITICAL | Closure capture overflow silently ignored | ✅ FIXED | eca0ee2 |
| CRITICAL-3 | CRITICAL | Module name collision undetected | ✅ FIXED | 6bda4b3 |
| CRITICAL-4 | CRITICAL | @checkpoint purity scanner blind to control-flow | ✅ FIXED | 9084a05 |
| MEDIUM-1 | MEDIUM | PTX register pool fixed-size silent overflow | ✅ FIXED | d56b230 |
| MEDIUM-2 | MEDIUM | Closure nesting error-node propagation | ⏳ OPEN |  |
| MEDIUM-3 | MEDIUM | use-decl path resolution deferred to runtime | ✅ FIXED | 6bda4b3 (bundled with CRITICAL-3) |
| MEDIUM-4 | MEDIUM | Reflection cell quota in bootstrap path | ✅ FIXED | ee5c37c |
| LOW-1 | LOW | PTX kernel symbols all point to same blob start | ⏳ OPEN |  |
| LOW-2 | LOW | Tile shape cap raises NotImplementedError not 91001 | ⏳ OPEN |  |
| LOW-3 | LOW | ckpt_is_pure accepts any AST_CALL as pure | ⏳ OPEN |  |

**8 of 11 resolved. 3 OPEN (1 MEDIUM, 2 LOW).**

---

## CRITICAL-1 — FFI_CALL missing from DCE SIDE_EFFECT_KINDS [FIXED 6bda4b3]

DCE silently drops void-return FFI calls (`puts(msg)`, `free(p)`, etc.). Fix: added FFI_CALL to SIDE_EFFECT_KINDS in dce.py.

## CRITICAL-2 — Closure capture overflow silently ignored [FIXED eca0ee2]

`cl_capture_tab_add_dedup` returns -1 on overflow but caller `mk_var_with_capture` discarded the return value. 5th+ free variable in a closure resolved to outer-scope vars or const_int(0). Fix: propagate -1 + emit trap 76002.

## CRITICAL-3 — Module name collision undetected [FIXED 6bda4b3]

`mod foo { fn bar }` mangled to `foo__bar`. If user also had top-level `fn foo__bar`, the second definition silently overwrote the first. Fix: collision check in flatten_modules.

## CRITICAL-4 — @checkpoint purity scanner blind to control-flow [FIXED 9084a05]

`ckpt_callees_pure` handled tags 2/3/4/5/9/16 but defaulted to "pure" for AST_IF (7), AST_LET (8), AST_WHILE (10), AST_SEQ (13). A `@checkpoint` fn with `if cond { call_impure() }` passed the gate; reverse-mode AD produced wrong gradients. Fix: explicit recursive dispatch added.

## MEDIUM-1 — PTX register pool overflow silent [FIXED d56b230]

`%r<32>` / `%rd<8>` declared pool size; counter unbounded; ptxas rejects on 33rd reg silently. Fix: bumped pools to 256 + per-prefix overflow check.

## MEDIUM-2 — Closure nesting error-node propagation [OPEN]

`parse_closure_lit` returns `mk_node(99, 76001, 0, 0)` on nested closure but tag-99 nodes aren't checked downstream. Eventually lower_ast.py defaults to const_int(0).

**Fix path**: lower_ast should raise on unknown AST tag, including tag 99. Or parser caller-sites should explicit-check.

## MEDIUM-3 — use-decl path resolution deferred [FIXED 6bda4b3]

`use foo::bar` registered alias without verifying foo::bar exists. Fix: post-flatten verification in flatten_modules.

## MEDIUM-4 — Reflection cell quota in bootstrap path [FIXED ee5c37c]

Python `lower_ast.py` raises on 65th Quote(). Bootstrap `bn_quote_bump_handle` lacked the cap check. Fix: cap moved INTO bn_quote_bump_handle so all callers benefit.

## LOW-1 — PTX kernel symbols same blob start [OPEN]

All `__helix_ptx_<name>` symbols defined before PTX bytes laid out, all resolve to start of PTX blob. Multi-kernel scenario relies on `.entry` name lookup at JIT time. If kernel names collide post-mangling, symbol table silently overwrites.

**Mitigated by CRITICAL-3 fix** (no name collisions now), so LOW.

## LOW-2 — Tile shape cap exception type [OPEN]

`_tile_cap_check` raises `NotImplementedError` instead of structured `HelixCompileError`. Trap 91001 reserved but never structured.

**Fix path**: introduce HelixCompileError or equivalent, replace NotImplementedError.

## LOW-3 — ckpt_is_pure accepts any AST_CALL [OPEN]

`ckpt_is_pure` (the secondary check on the @checkpoint fn's own body) accepts AST_CALL (tag 16) by checking only arguments, not callee body. A `@checkpoint fn f() { impure_fn(x + 1) }` passes the gate.

**Fix path**: walk callee body for purity.

---

This audit was originally completed by a subagent on 2026-05-08; the inline findings text was preserved in the project's session transcript and reconstructed into this doc on 2026-05-10 during Stage 28.8 cycle 1 (the cycle 1 silent-failure-hunter audit flagged the missing doc).
