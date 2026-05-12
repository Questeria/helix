# Audit Stage 28.9 cycle 75 — Silent failures

**Scope.** Read-only HEAD `9a51cbf`. Narrow conservative scope. Prior C1–C74 + deferred-known not re-flagged.
**Criterion.** 0 findings at conf >=75%.

## Result: 0 findings at >=75% — PASS

The cycle-74 fix-sweep (commit `9a51cbf`, "Stage 28.9 cycle-74 fix-sweep: cycle-73 type-design CN-1 (double-descent)") modifies a single source file (`helixc/frontend/totality.py`) plus a regression test in `helixc/tests/test_deprecated.py`. The narrow-scope re-audit focused on:

1. The cycle-74 totality double-descent fix itself.
2. New silent-failure regressions in the cycle-66..68 fix-sweep surface (`flatten_modules.py` intra-mod-alias rewriting + ImplBlock arm).
3. `match_lower` and `pytree` per the prompt.

No silent-failure findings at confidence >= 75% were uncovered in this surface.

---

## Cycle-74 fix verification (totality double-descent)

**Source.** `helixc/frontend/totality.py` lines 64-76:

```python
def visit_Call(self, node: A.Call) -> None:
    callee = node.callee
    if isinstance(callee, A.Name) and callee.name == self.fn_name:
        self.calls.append(node)
    # cycle-74 comment: do NOT call self.generic_visit(node) here.
    # ASTVisitor.visit auto-descends AFTER this override returns
    # unless we return False.
```

**Semantics check against `ast_walker.ASTVisitor.visit`** (lines 180-197):
- `result = method(node)` runs the override (which returns `None`).
- The post-condition `if result is False: return result` does NOT trigger (`None is False` is False in Python).
- `self.generic_visit(node)` is then called by the base class exactly ONCE.

**Empirical probes** (run against HEAD `9a51cbf`):

| Probe | Source | Expected | Observed |
|-------|--------|----------|----------|
| Nested self-call (cycle-74 test) | `fn rec(n) { rec(rec(n-1)) }` | 2 | 2 |
| Triple-nested self-call | `fn rec(n) { rec(rec(rec(n-1))) }` | 3 | 3 |
| If-branch self-call | `fn rec(n) { if n > 0 { rec(n-1) } else { 0 } }` | 1 | 1 |
| Match-arm self-call | `fn rec(n) { match n { 0 => 0, _ => rec(n-1) } }` | 1 | 1 |
| ImplBlock-in-ModBlock recursion (nested item walk) | `mod outer { impl Foo { fn bad(n) { bad(n) } } }` | flagged | flagged ("recursive but no parameter strictly decreases") |
| ModBlock-in-ModBlock recursion | `mod a { mod b { fn bad(n) { bad(n) } } }` | flagged | flagged |
| ImplBlock-in-ModBlock total recursion | `mod outer { impl Foo { fn good(n) { if n>0 { good(n-1) } else {0} } } }` | not flagged | not flagged |

All probes match expected counts/verdicts. The pre-cycle-74 double-descent (which would have produced 4 for the first probe and 9 for the triple-nested probe) is not present at HEAD `9a51cbf`.

`iter_fn_decls` (ast_walker.py:214-274) is the FnDecl-enumeration surface used by `check_totality`. The recursion arms cover `FnDecl`, `ImplBlock.methods` (direct only — sufficient because the parser does not produce ImplBlock-in-ImplBlock), and `ModBlock.items` (recursive — handles ModBlock-in-ModBlock and ImplBlock-in-ModBlock via the recursive `_walk`). No ImplBlock-in-ImplBlock pattern is reachable from the parser surface (parser.py:174 only parses ImplBlock as a top-level/mod-level item).

`test_totality` (15 tests) and `test_deprecated` (28 tests, including the new `test_c73_cn1_totality_no_double_descent`) all pass at HEAD `9a51cbf`.

---

## flatten_modules cycle-66/68 fix-sweep surface — clean

The cycle-66 + cycle-68 + cycle-71 changes to `helixc/frontend/flatten_modules.py` introduce:

- `intra_mod_aliases` dict built BEFORE the lift loop (lines 108-112), keyed only on `FnDecl/StructDecl/EnumDecl/ConstDecl/TypeAlias` siblings — explicitly excludes `ModBlock` and `AgentDecl` (cycle-68 CN-1 fix). Verified: a nested `mod inner { ... }` does NOT register an alias, so a sibling `Name("inner")` reference is not rewritten to a non-existent `outer__inner`.
- Per-direct-lift index list `local_lift_indices` (lines 126, 139) replacing the pre-cycle-68 range-based walk. Verified: recursive `_flatten_one` calls for nested ModBlocks do NOT bleed outer-scope aliases into inner-scope item bodies because the outer call's post-loop walk only touches indices it recorded directly.
- ImplBlock arm (lines 184-210, cycle-68 CN-2 fix) that mangles `sub.target` against `intra_mod_aliases` and rewrites every method body via `_rewrite_expr(m.body, intra_mod_aliases)`. Verified: at the time the ImplBlock arm runs, `intra_mod_aliases` is fully populated (built in the pre-loop pass at 109-112), so the rewriter sees a complete alias dict.
- StructLit.name remap (lines 292-304, cycle-71 CN-3) via the same alias dict.

The `_rewrite_expr` catchall (lines 356-360) raises `NotImplementedError` on unhandled Expr subclasses — a loud-fail discipline established cycle-57. No silent-failure regression visible in the cycle-66..68 + cycle-71 delta.

The `_flatten_one` ImplBlock arm intentionally does NOT increment `n` (the count of lifted items). This is consistent: the ImplBlock is replaced in place, not lifted to top level with a mangled prefix-name. The post-flatten driver invokes `flatten_impls` next, which performs the actual method-to-top-level lifting. Examined: not a silent-failure pattern.

---

## match_lower & pytree — out of cycle-66..68 delta

- `match_lower.py`: last touched in cycle-60 (commit `475083f`). No cycle-66..68 churn. Out of scope per "new regressions" criterion.
- `pytree.py`: last touched in cycle-58 (commit `722baf8`). No cycle-66..68 churn. Out of scope.

Both files were checked for last-modification timestamp via `git log` only; no defect probes run since the cycle-66..68 fix-sweeps did not touch them.

`test_pytree` (24 tests), `test_match` (full module), and `test_ast_walker` all pass at HEAD `9a51cbf`.

---

## Sub-threshold notes (conf <75)

- **N-1** (conf 55): `flatten_modules._flatten_one` ImplBlock arm at line 204 (`m.body = _rewrite_expr(m.body, intra_mod_aliases)`) mutates the original method object in place, then wraps the (mutated) `m` reference in a fresh `ImplBlock` at line 206. The rewrite is functionally correct because the AST is otherwise expected to be tree-shaped (no aliasing of FnDecl nodes between ImplBlocks). If a future macro-expansion pass produced two ImplBlocks sharing a method-node reference, the first ImplBlock's rewrite would silently mutate the second. No current pipeline produces such sharing; flagging for future-pass awareness only. Below threshold because no realistic surface exercises it.

- **N-2** (conf 50): `_SelfCallCollector` extends `ASTVisitor` but the `visit_Call` override has no explicit `return None` — it falls off the end. Python's implicit-None return matches the base class's "anything-not-False → auto-descend" semantics. Stylistic inconsistency with `panic_pass._PanicCollector.visit_Call` (which also falls-off-end). Below threshold because behavior is correct.

---

## Findings NOT re-flagged (deferred-known, per prompt scope)

- `monomorphize._mangle_ty` silent catchall — promoted to `raise NotImplementedError` in cycle-71 fix-sweep (commit `a22cba0`), per cycle-74 commit message clarification. No longer a finding.
- `hash_cons._ast_equal` SHA-256 fallback — genuinely deferred-known.
- `typecheck` pre-flatten in `check.py` — deferred per prompt (broader pipeline asymmetry).
- `struct_mono` pre-flatten in `check.py` — deferred per prompt.
- `autotune.collect_autotuned_fns` missing `iter_fn_decls` — deferred per prompt.

## Edits made

NONE. This audit was conducted in STRICT READ-ONLY mode. No source files were modified. The only Write call was for this audit document at `docs/audit-stage28-9-cycle75-silent-failures.md`.

## Files inspected

- `C:/Projects/Kovostov-Native/helixc/frontend/totality.py` (full file)
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_walker.py` (full file)
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_modules.py` (lines 1-399)
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py` (ImplBlock at 545-557)
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_impls.py` (lines 1-160 — call-rewrite arm)
- `C:/Projects/Kovostov-Native/helixc/check.py` (flatten ordering at 431-471)
- `C:/Projects/Kovostov-Native/helixc/backend/x86_64.py` (flatten ordering at 3072-3107)
- Git history of `match_lower.py` and `pytree.py` (last modified cycles).
- Cycle-74 commit `9a51cbf` diff (full).

Probes run against the installed HEAD via direct `python -c` invocation; `test_totality` (15/15), `test_deprecated` (28/28), `test_pytree + test_match + test_ast_walker` (65/65) all pass.
