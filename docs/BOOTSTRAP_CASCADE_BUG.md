# Bootstrap Cascade-Depth Self-Host Bug

**Status:** **RESOLVED 2026-05-07.** Root cause: `read_file_to_arena`
buffer (256 KB) was being silently overrun by the bootstrap source
(~261 KB at the time of resolution, with < 1 KB margin). Each new fn
or @pure helper added to kovc.hx pushed the concatenated source past
the buffer; K1's read truncated; the truncated source was missing
tail-end fns; K1 produced K2 whose call sites ud2-patched on missing
symbols; K2 SIGILLed on first call. Mis-attributed for weeks as a
"cascade-depth bug" because the symptom was correlated with adding
arms to deep cascade fns (those happen to add the most bytes per
edit). The fix: bump BUF_SIZE 256 KB -> 1 MB in both
helixc/backend/x86_64.py (Python emit) and the four uses in
helixc/bootstrap/kovc.hx's emit_read_file_to_arena_body. Kept in
lock-step. Stage 2.4b MUL+ / 2.5b/c / 1.5 / 3+ all unblocked.

The narrative below is preserved as the historical record of how the
bug was characterized and eventually resolved.

---

## Original status (preserved)
**Was:** Open. Blocks Stage 2.5b+ and Stage 2.4b MUL/DIV/MOD/comparisons.

## Symptom

Adding any new `} else { if t == X { ... }` arm to certain large
cascade fns in `helixc/bootstrap/kovc.hx` or `helixc/bootstrap/parser.hx`
causes K2 self-host to fail with either SIGILL or infinite-loop timeout,
**even when the new arm is dead code** (e.g., the lexer never produces
the matching token tag).

## Affected fns

The "big cascade fns" exhibiting this behavior:

- `parse_primary` (helixc/bootstrap/parser.hx) — ~30+ arms in a
  `} else { if t == X { ... }` chain.
- `emit_ast_code` (helixc/bootstrap/kovc.hx) — ~35+ arms.
- `expr_type` (helixc/bootstrap/kovc.hx) — ~24 arms.

Adding an arm to ANY of these breaks self-host idempotence past the
current threshold (Stage 2.4b ADD+SUB landed; Stage 2.4b MUL did not).

## What's safe (does NOT break)

Confirmed via 9+ probe experiments:

- **Top-level fn additions**: `fn _dummy() -> i32 { 0 }` at file scope.
- **Lets inside existing fns**: `let _x = 0;`.
- **`if X { ... }` blocks inside fns (with or without mutation)**:
  including `if b2 == 56 { p = p + 3; }` in lex_int.
- **Lexer changes**: e.g., adding a new token tag and lex flag.
- **Parser type-ident parsing**: editing the `b0/b1/b2` byte-comparison
  logic in `parse_param_type` / `parse_fn_decl ret_ty`.

## What breaks

- Adding a new arm to `parse_primary`, `emit_ast_code`, or `expr_type`.
- Even pure dead code (e.g., `} else { if t == 99 { ... }` where
  TK 99 is never produced) triggers the failure.
- **NEW (probe 10, 2026-05-07):** Adding a NEW `@pure` top-level fn
  before `expr_type` (with no callers, no side effects, just a small
  cascade returning literal type tags) ALSO triggers SIGILL in K2.
  This is stronger than previously believed: the bug can be tripped
  WITHOUT modifying any cascade — simply growing the kovc.hx source
  past a threshold breaks self-host. Disproves "must modify cascade"
  as a necessary condition.

## Failure modes

Two distinct failure modes observed:

1. **K2 SIGILL (exit 132)**: K2 binary contains a reachable `ud2`
   instruction. Observed locations include `0x1c0e6` and `0x1c075`,
   each sitting before some fn's epilogue (`mov rsp, rbp; pop rbp`
   without preceding `jmp +5` jump-over).

2. **K2 timeout (subprocess hangs >30s)**: K2 binary enters an infinite
   loop. Specific to lex_int-related changes in some configurations.

## Hypothesis pool

- **AST_IF byte-count math off-by-N at depth**: the
  `n_cond + n_test + 6 + n_then + 5 + n_else` formula in
  `emit_ast_code` t==7. Inspected: looks correct.
- **patch_rel32 backpatching bug at depth**: rel32 displacements have
  ±2GB range, fns are well under. Unlikely to be the cause.
- **Stack frame allocation**: prologue allocates 1024 bytes (128 slots).
  Each AST_IF emit recurses, but only adds ~3-4 lets per level. Bumping
  cap to 256 + prologue 4096 ALSO breaks self-host on its own — so cap
  is not the cause.
- **Arena layout shift**: bind_state init pushes `N` zero slots; growing
  `N` shifts all subsequent allocations including the ELF base. Some
  hardcoded reference might assume original layout. Not yet ruled out.
- **Source-size threshold in Python helixc**: maybe Python's backend
  has a buffer that overflows when kovc.hx grows past a specific size.
  **Strongly elevated by probe 10**: adding a fn that doesn't touch
  any cascade still breaks self-host, suggesting the issue is sensitive
  to total source size / total emitted-code size, not just cascade-arm
  count.
- **rel32 placeholder slot pool exhaustion**: helixc-Python has a finite
  pool of forward-jump rel32 placeholders. Adding any new fn that emits
  conditional branches (each `if` lowers to ~2 placeholders) eats slots.
  If the pool wraps without proper book-keeping, an old jump's
  displacement could be overwritten by a newer jump. Worth checking
  helixc/backend/elf.py and codegen.py for any such bounded pool.

## Workarounds

- **Lexer-side changes**: SAFE. Stage 2.5a (`_i8` suffix detection)
  landed via this path.
- **Refactor cascade fns to table-driven dispatch**: would side-step
  the cascade-arm pattern entirely. Big refactor; deferred.

## Reproducer (minimal)

Apply this patch, run `python -m pytest helixc/tests/test_codegen.py::test_bootstrap_kovc_self_host_loop -q`:

```diff
@@ parser.hx, parse_primary, near other arms @@
     } else { if t == 36 {
         ...
         mk_node(38, v, 0, 0)
+    } else { if t == 99 {  // dead code, TK 99 never produced
+        let v = tok_p1(tok_base, k);
+        cur_advance(sb);
+        mk_node(99, v, 0, 0)
     } else { if t == 25 {
@@ end of parse_primary @@
-    }}}}}}}}}}
+    }}}}}}}}}}}
```

Result: self-host test fails with K2 SIGILL or timeout.

## Next steps

1. Byte-diff K1-baseline vs K1-with-dead-arm to find which Python
   helixc emission differs. The diff should pinpoint the buggy
   instruction.
2. Or, refactor expr_type/parse_primary/emit_ast_code to flatter
   dispatch (perhaps a switch-like construct or table lookup) that
   avoids the deep cascade pattern entirely.
3. Once unblocked, resume Stage 2.4b MUL/DIV/MOD/comparisons,
   Stage 2.5b parser+codegen for i8, and onward.

## Currently-blocked stages

- Stage 2.4b (MUL/DIV/MOD/LT/GT/LE/GE u64 dispatch)
- Stage 2.5b (i8 parser+codegen+expr_type)
- Stage 2.5c (i16, u16, narrow load/store)
- Stage 1.5 (bf16/f16) — independently doable, but defer until cascade
  bug is fixed since bf16/f16 will need new expr_type arms too
- Stage 3 onwards (Strings, Tuples, Structs, Enums, etc.) — all need
  expr_type additions

Stage 2.4b ADD+SUB and Stage 2.5a (lex-only) are the latest landings.
