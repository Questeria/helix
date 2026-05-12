# Audit Stage 28.9 cycle 88 — Silent failures

Scope HEAD e096767 (Stage 28.9 cycle-87 audit clean).

Rotated files (NARROW conservative scope):
- helixc/ir/passes/fdce.py — function-level DCE edge cases
- helixc/frontend/parser.py — parser silent fallthroughs in rare-path rules
- helixc/backend/elf_dyn.py — dynamic-link emission for new FFI paths

Deferred-known NOT re-flagged (per cycle-87 deferral list):
- monomorphize._mangle_ty, hash_cons._ast_equal silent catchalls
- typecheck/struct_mono pre-flatten in check.py
- autotune.collect_autotuned_fns missing iter_fn_decls
- struct_mono.mangle_struct collision sub-75

Prior C1-C87 findings NOT re-flagged.

## Methodology

1. Confirmed HEAD via `git log --oneline -2` (e096767, d8e5807 — no code change since cycle-86 fix).
2. Read each rotated file in full or via targeted Grep.
3. Searched for: silent `except: pass/continue/return`, `return None` without caller-guard, `self.i += 1` skip-on-unknown without EOF guard, lenient token-eat patterns without diagnostic, and missing-input default-return without warning.
4. Cross-referenced docstrings to distinguish intentional lenient behaviour from silent failure.

## Findings

### fdce.py
- `entry_fn not in module.functions -> return 0`: explicitly documented intent ("we don't want to silently empty the module"). Not silent.
- QUOTE ast_pretty scan over-approximates reachability via identifier regex — conservative (over-keeps, never under-keeps). Cannot silently drop a live fn.
- Pub-prefix heuristic widened to `fn.attrs.get("is_pub")` and `kernel` — correctly handled, no silent gap.
- No silent-failure findings.

### parser.py
- `_parse_trait_decl` (lines 215-240): trait bodies are explicitly stubs ("Phase 1.8: traits are accepted but only as documentation"). The skip-unknown-token-fallthrough (`else: self.i += 1`, line 239) and the trait-fn param skip-loop (lines 223-227) lack explicit `T.EOF` guards. Worst case on a malformed trait body with unmatched braces/parens: parser hangs in skip-loop instead of raising ParseError. However: (a) traits are parser-stub-only and discarded, (b) `_peek` clamps to `toks[-1]` so behaviour is bounded by EOF-token presence + downstream `_eat(T.RBRACE)` failure mode, (c) this construct predates Stage 28.9 rotation and is not a regression. Conf < 75% as a Stage 28.9 silent-failure; flagged here for visibility but NOT counted.
- `_parse_string_attr_arg` (line 388): documented "lenient" skip-to-RPAREN — intentional, EOF-guarded.
- `@attr(...)` arg scan (lines 300-321): documented "Skip non-ident tokens" — intentional, depth-balanced, EOF-guarded via `if t.kind == T.EOF: raise ParseError("unclosed attribute args", t)` at line 305-306.
- `_merge_stdlib` missing-file path (lines 1582-1589): NOT silent — emits stderr warning and respects `HELIXC_STDLIB_STRICT` env (Audit 28.8 A8 fix); documented.
- No silent-failure findings >= 75% conf.

### elf_dyn.py
- All region offsets/sizes are assert-checked (`assert len(out) == layout.X_offset` × 9). A mismatch raises AssertionError, not silent.
- `plan_layout` raises RuntimeError when phdrs+interp exceed CODE_OFFSET (line 217-219). Not silent.
- `DynLinkInfo.add_import` is idempotent and deterministic — no silent collision.
- Single PT_LOAD R+W+X is documented Phase-0 simplification, not a silent gap.
- Dynamic-table entry count `n_dyn_entries = len(dyn.needed_libs) + 12` matches the 12 fixed entries appended (DT_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_SYMENT, DT_PLTGOT, DT_PLTRELSZ, DT_PLTREL, DT_JMPREL, DT_FLAGS, DT_FLAGS_1, DT_NULL) and is assert-verified at line 344.
- `.hash` SYSV table: `nchain = n_syms` and chain construction handles `n_syms == 1` (no imports) correctly: `bucket_vals=[0]`, `chain_vals=[0]`, loop `range(1, 0)` is empty, `if n_syms > 1` guard skipped — produces a valid empty-symtab hash table.
- No silent-failure findings.

## Result

**PASS** — 0 findings at conf >= 75%.

## No edits

This audit performed read-only inspection. Only this audit document was written. No source file in `helixc/` was edited.
