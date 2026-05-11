# Stage 28.8 Cycle 18 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit (HEAD)**: 0243d5c — "Phase A staging refined: insert
28.8.1 (determinism) + 28.8.2 (walker lib)".

**Note on HEAD movement vs cycle 17**: one new commit landed at
0243d5c between cycle 17 (c6136d4) and cycle 18. `git show --stat
0243d5c` shows the diff is entirely doc-only: +1163/-17 lines
across two markdown files (`docs/APPROACH_A_PLAN.md` and
`docs/helix-pre-phase-A-finalization-research.md`). The commit
message classifies the change as "Phase A staging refined" —
plan-document refresh, not production-code change.

`git diff c6136d4..HEAD -- 'helixc/' '*.py' '*.hx'` confirms:
**zero production-code surface delta** vs cycle 17 HEAD. Every
clean verdict on cycle 17 holds on cycle 18 unless the fresh-eyes
re-walk surfaces an overlooked window. The new commit could
NOT have opened a silent-failure window in production code by
construction (it only touches `docs/*.md`).

**Scope**: any silent-failure window NOT already counted in
cycles 1-17 as a carryover. Documented carryovers (audit-C4-1
CRITICAL, audit-C4-4 HIGH, audit-C4-8 LOW, C5-10 LOW,
monomorphize_safe docstring drift, D-vs-Quote diagnostic text,
C7-1 test-coverage gap) are NOT re-flagged per the user's
strict re-flag rule (a carryover is re-flagged only if it
CHANGED since the prior cycle — and none did, because no
production code changed since cycle 17).

**Strict criterion** (per user directive 2026-05-10): cycle
counts CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Findings already in the carryover
ledger are explicitly excluded.

**Clean-counter state going into cycle 18**: 1/5 (counter reset
to 0 by C16-1, cycle 17 = 1/5). Cycle 18 is the second
clean-cycle attempt of the post-C16-1-fix re-accumulated window.

---

## Method

1. **Read cycle-16 + cycle-17 silent-failure verdicts**: both
   CLEAN for the silent-failures lens. Cycle 17 fix-sweep closed
   C16-1 (HIGH, wide-array-elem silent truncation) by adding the
   `_check_array_elem_size_supported` helper to x86_64.py +
   wiring it into LOAD_ELEM / STORE_ELEM emit sites.
2. **Confirmed HEAD delta vs cycle 17** is doc-only:
   - `git log --oneline c6136d4..HEAD` → `0243d5c Phase A
     staging refined: insert 28.8.1 (determinism) + 28.8.2
     (walker lib)`.
   - `git show --stat 0243d5c` → only `docs/APPROACH_A_PLAN.md`
     and `docs/helix-pre-phase-A-finalization-research.md`
     touched (+1163/-17 lines).
   - `git diff c6136d4..HEAD -- 'helixc/' '*.py' '*.hx'` returns
     empty. **Zero production-code surface delta** vs cycle 17
     HEAD.
3. **Verified the cycle-17 fix at HEAD**
   (`helixc/backend/x86_64.py:983-1003` helper +
   `helixc/backend/x86_64.py:2738-2776` LOAD_ELEM/STORE_ELEM
   call sites): all three sites unchanged from cycle-17 audit.
   No production change at this site since the fix landed.
4. **Cycle-18 fresh-eyes rotation**: per the user directive,
   pick a NEW least-covered module not rotated for ≥3 cycles.
   Cycle 16 covered `effect_check.py` + `totality.py` + `cse.py`
   line-by-line. Cycle 17 was the fix-sweep audit and didn't
   rotate. Per the user's suggested candidates, the rotation
   for cycle 18 is:
   - **`helixc/ir/passes/fdce.py`** — full-module audit (only
     previously enumerated as "call-graph source check vs
     TRACE_*" in the stability table; never line-by-line).
   - **`helixc/frontend/hash_cons.py`** interior (lines 82-342)
     — only line 335 (`raise HashConsError`) previously
     enumerated; the `_Sharer` class internals + `_ast_equal`
     +  `_stmt_equal` interior never line-by-line.
   - The new commit's diff in `docs/*.md` — out-of-scope by
     construction (doc-only). Skim only.
   - Note: `effect_check.py` lives at `helixc/ir/passes/`, not
     `helixc/frontend/` — cycle 16 audited the correct module;
     the user's prompt's path was approximate. Re-confirmed
     the same file is the only effect_check.py in the tree.
5. **Read-only**: no edits to production code or tests.

---

## Fresh-eyes walk for cycle 18

### `helixc/ir/passes/fdce.py` — full-module audit

**File invariants**:
- 0 `try` blocks.
- 0 `except` arms.
- 0 `raise` sites.
- 4 `.get(...)` calls (lines 49, 53, 57, 70, 72, 82 — see
  individual walks).
- 1 early-return short-circuit (lines 30-31).

**Surface walk** (line numbers per the file at HEAD 0243d5c,
which is identical to cycle 17 HEAD c6136d4 for this file):

1. **Module docstring (lines 1-20)** — documents the algorithm
   and the explicit "Skips removal if `entry_fn` is missing —
   we don't want to silently empty the module" guarantee
   (lines 16-17). The documented contract is correct: the
   short-circuit is the only way to avoid silently deleting
   every fn when entry_fn is misspelled.
2. **Entry-fn missing short-circuit (lines 30-31)**:
   ```python
   if entry_fn not in module.functions:
       return 0
   ```
   `return 0` reports zero fns dropped — the count interface
   to the caller is preserved. **Is this a silent fallback?**
   The caller in `check.py:580` invokes `fdce_module(mod)`
   for cleanup after grad_pass; the module-level main-fn check
   is performed upstream by typecheck's `_check_main_signature`
   (which raises a TypeError_ for missing main with full
   user-facing diagnostic). Similarly, `compile_module_to_elf`
   at backend-link time requires main and raises LOUDLY on
   missing main. The fdce_module short-circuit is therefore
   correct: the missing-entry-fn case is LOUD upstream
   (typecheck) AND LOUD downstream (link-fixup raises
   `ValueError("unresolved symbol: ...")`). The short-circuit
   prevents the silently-empty-module miscompile that would
   occur if fdce ran without a root. Not a silent-failure
   window. Confirmed non-finding.
3. **CALL target resolution (lines 48-51)**:
   ```python
   if op.kind == tir.OpKind.CALL:
       target = op.attrs.get("target")
       if isinstance(target, str):
           called.add(target)
   ```
   `op.attrs.get("target")` defaults to `None` if missing; the
   `isinstance(target, str)` guard silently skips non-string
   or absent targets. **Is this a silent-failure window?** A
   CALL op without a `target` attribute would be a malformed
   IR; the backend's CALL emit site at `x86_64.py:1665`
   (`target = op.attrs.get("target", "?")` + downstream
   `call_rel32(str(target))`) would emit a fixup with target
   `"?"`, which fails at `Buffer.patch()` line 120-121 with
   `raise ValueError(f"unresolved symbol: {f.target}")`. So
   fdce's silent-skip is safe by construction — the downstream
   linker is the authoritative source-of-truth for catching
   malformed CALL. fdce treating it as "callee-less" doesn't
   hide the bug. Confirmed non-finding.
4. **MODIFY verifier_fn (lines 52-55)** — same pattern as
   CALL target. The `verifier_fn` attr is consumed downstream
   by `x86_64.py:2723` (`self.asm.call_rel32(verifier_name)`)
   which produces a fixup. If the verifier_fn is missing, the
   call would still be emitted to a `<unknown>` target name
   which fails at link time. Confirmed non-finding.
5. **QUOTE ast_pretty regex (lines 56-61)**:
   ```python
   elif op.kind == tir.OpKind.QUOTE:
       pretty = op.attrs.get("ast_pretty", "")
       if isinstance(pretty, str) and pretty:
           for ident in _ID_RE.findall(pretty):
               if ident in all_fn_names:
                   called.add(ident)
   ```
   `op.attrs.get("ast_pretty", "")` defaults to empty string;
   the `if isinstance(pretty, str) and pretty:` guard silently
   skips empty or non-string. **Is this a silent-failure
   window?** A QUOTE op's only producer is
   `lower_ast.py:2128-2131` which always sets `ast_pretty=
   _pretty(expr.inner)`. So in production, `ast_pretty` is
   always present as a non-empty string. The `.get` default
   is defense-in-depth for test-side hand-built modules. If
   the default fired in production, the fdce conservative-
   include would skip ALL callees from this QUOTE op — which
   could result in **dropping a fn reachable only via splice**.
   That would be a miscompile if it could happen.

   **However**: lower_ast.py is the only QUOTE producer in
   the tree (verified by `grep ast_pretty helixc/` returning
   only lower_ast.py + fdce.py + nothing else). So
   `ast_pretty` is always set when reaching fdce. The
   conservative empty-string fallback is correct
   defense-in-depth without a live miscompile window. The
   `_ID_RE` regex is over-inclusive (catches non-fn-name
   identifiers from the quoted expression body), but the
   `if ident in all_fn_names` guard at line 60 narrows back
   to actual module fn names, which is correct. Confirmed
   non-finding.
6. **Roots — is_pub + kernel attrs (lines 69-73)**:
   ```python
   for name, fn in module.functions.items():
       if fn.attrs.get("is_pub"):
           worklist.append(name)
       elif fn.attrs.get("kernel"):
           worklist.append(name)
   ```
   `fn.attrs.get("is_pub")` defaults to None (treated as
   falsy). If a fn were `is_pub=True` but the attr were
   misspelled as `pub`, the guard would not pick it up. That
   is a **producer-side bug** in the attribute-name canon
   — the author must use the canonical name `is_pub`. Not a
   runtime silent failure; static-check-at-authoring-time
   hazard. Confirmed non-finding.
7. **Live-set fixpoint (lines 75-84)**:
   ```python
   while worklist:
       n = worklist.pop()
       if n in live:
           continue
       if n not in module.functions:
           continue
       live.add(n)
       for c in callees.get(n, ()):
           if c not in live:
               worklist.append(c)
   ```
   `if n not in module.functions: continue` (lines 79-80)
   silently skips worklist entries that aren't in module
   functions. This can happen if `callees` contains
   identifiers from QUOTE.ast_pretty's regex that matched a
   var name shadowing a not-quite-a-real-fn name. Wait —
   the QUOTE regex at line 60 already narrows to
   `if ident in all_fn_names` (which is the set of module
   fn names), so anything in `callees[n]` IS a real module
   fn name... or was at fdce-entry time.

   But `callees.get(n, ())` (line 82) returns the empty
   tuple if `n` is not a key in `callees`. The `callees`
   dict is built from `module.functions.items()` at line 44,
   so every fn in module.functions has an entry. The `.get`
   default is defensive against external callees pulled in
   via QUOTE regex (which were narrowed to all_fn_names
   anyway). The narrowing IS done at line 60. Both the
   line-79 `if n not in module.functions: continue` and the
   line-82 `callees.get(n, ())` default are belt-and-
   suspenders defense — neither hides a silent failure.
   Confirmed non-finding.
8. **Dead-fn deletion (lines 87-89)**:
   ```python
   dead = [n for n in module.functions if n not in live]
   for n in dead:
       del module.functions[n]
   return len(dead)
   ```
   Returns the dropped count to the caller. The caller in
   `check.py` doesn't currently print this count
   (verified by reading check.py:580 — just `fdce_module(mod)`
   with no count display). **Is non-display silent?**
   No — the function's contract is to remove dead fns;
   the caller doesn't NEED to see the count. The backend
   CLI at `x86_64.py:3181` also doesn't display
   (`f_removed = fdce_module(mod)` then `f_removed` is
   never printed). Both call sites correctly use the
   pass for its side effect. The return-value is for
   testing (`helixc/tests/test_fdce.py` exercises the
   count). Not a silent-failure window.

**Conclusion**: `fdce.py` has zero exception handlers, zero
raise sites, and four `.get(...)` calls — each justified
either by upstream-producer-guarantees (lower_ast is the
sole QUOTE producer with `ast_pretty` always set) or by
downstream-linker-LOUD-failure-mode (CALL target unresolved
fires `ValueError` at `Buffer.patch()`). The entry_fn
short-circuit at line 30-31 is the documented "don't
silently empty the module" guard. Confirmed non-finding for
cycle 18.

### `helixc/frontend/hash_cons.py` — interior walk (lines 82-342)

Cycle 13 already covered `helixc/frontend/hash_cons.py:335
raise HashConsError` (the trap-20001 collision raise). The
INTERIOR (`hash_cons` entry, `_ast_equal`, `_stmt_equal`,
`_Sharer.share_in`, `_share_stmt`, `_maybe_share`) was never
line-by-line audited in any prior cycle. Cycle 18 covers it
for the first time.

**File invariants**:
- 0 `try` blocks.
- 0 `except` arms.
- 1 `raise` site (line 335 — trap-20001 collision; already
  audited cycle 13 clean).
- 0 `.get(...)` calls on AST nodes (the `_canon.get(h)` at
  line 329 is a dict lookup, not an attribute access).
- 16 `_maybe_share(node)` recursion sites across the
  `_Sharer` class.

**Surface walk**:
1. **`HashConsError` class (lines 57-59)** — typed exception
   with `trap_id = 20001`. Already audited cycle 13 clean.
   Stable non-finding.
2. **`_SHAREABLE` tuple (lines 68-79)** — 13-element type
   tuple defining which AST node classes are shareable. The
   comment at lines 64-67 documents the conservatism:
   "Expressions that are pure-functions of their children
   ... no implicit binders, no side-channel state ... Block/
   Match/For/While/Loop out — their semantics depend on
   scope and execution order". Any new AST class added to
   the parser MUST be evaluated for inclusion at design
   time. **Static-check-at-authoring-time hazard**; if a
   new pure-expression class is added but not added to
   `_SHAREABLE`, sharing simply doesn't apply (a missed
   optimization, not a miscompile). Not a silent-failure
   window. Confirmed non-finding.
3. **`hash_cons(prog)` entry (lines 82-89)**:
   ```python
   def hash_cons(prog: A.Program) -> int:
       sharer = _Sharer()
       for item in prog.items:
           if isinstance(item, A.FnDecl):
               sharer.share_in(item.body)
       return sharer.merged
   ```
   The `isinstance(item, A.FnDecl)` filter silently skips
   non-FnDecl items (StructDecl, EnumDecl, ModBlock, ImplBlock,
   UseDecl, ConstDecl, TypeAlias, ModuleDecl). **Is this a
   silent-failure window?**

   Skipping StructDecl, EnumDecl, ConstDecl, TypeAlias,
   UseDecl: these decls have no expression bodies (constants
   have a single value that's already a literal; type aliases
   are type-level only). Correctly skipped.

   Skipping ModBlock: parser produces `ModBlock(name="foo",
   items=[FnDecl(...)])` for `mod foo { fn bar() { ... } }`.
   The FnDecls inside the ModBlock are NOT walked by
   `hash_cons`. Hash-cons therefore misses dedup
   opportunities inside mod blocks. **Is this a miscompile
   or a missed optimization?**

   - In `check.py:542`, hash_cons is called BEFORE
     `lower(prog)` at check.py:565, but check.py never
     calls `flatten_modules`. ModBlock items propagate to
     lower(), which itself only walks `A.FnDecl` items at
     `lower_ast.py:181-186`. So mod-block fns are dropped
     from the IR module entirely — they never see hash_cons
     OR lower. The downstream link-fixup at
     `x86_64.py:Buffer.patch()` lines 120-121 raises
     `ValueError("unresolved symbol: foo::bar")` LOUDLY if
     main calls `foo::bar()`.
   - In `x86_64.py:3151`, hash_cons is called AFTER
     `flatten_modules` at x86_64.py:3090 → at this point
     the ModBlock items have already been lifted to top-
     level FnDecls. Hash-cons walks the lifted FnDecls
     correctly.
   - In `check.py:542`, hash_cons being called WITHOUT
     prior `flatten_modules` means mod-block FnDecls are
     not hash-consed. The function returns a sharing-count
     that under-counts the true potential. **However**:
     the user-visible output `print(f"   hash-cons:
     {n_shared} AST node(s) deduped")` (check.py:543)
     reflects the actual work done. The user is not
     promised that hash-cons covers mod-block contents
     because the surrounding pipeline (which check.py
     drives) never lifts those contents — they're dropped
     downstream too. The user sees a downstream link error
     ("unresolved symbol"), not a silent miscompile.

   The chain is **LOUD downstream**, so hash_cons skipping
   mod-block contents is **structurally consistent with the
   rest of check.py's pipeline** — both miss flatten_modules.
   Hash_cons by itself is not a silent-failure window; the
   underlying issue is **check.py's missing flatten_modules
   stage**, which is a known pipeline-completeness gap
   that surfaces LOUDLY at the linker step. This concern is
   in the same family as carryover **audit-C4-8** (check.py
   doesn't call fn-mono — pipeline-stage omission that
   surfaces LOUDLY downstream). Filing as a **deferred
   observation** below, NOT as a new silent-failure finding,
   because:
   - The downstream failure mode is LOUD
     (`unresolved symbol` at Buffer.patch).
   - The check.py-without-flatten_modules concern is
     structural and predates the cycle-window (going back
     to the original check.py design at commit `0a21d41`).
   - The user-facing UX could be improved (a better
     diagnostic than `internal error: unresolved symbol`),
     but that's a **code-review-lens** concern about
     diagnostic quality, not a silent-failures finding
     about hidden error.
   - Re-flagging would violate the strict re-flag rule
     because this is the same shape as the audit-C4-8
     carryover (pipeline-stage omission, downstream-LOUD).

   Confirmed non-finding for the silent-failures lens.
4. **`_ast_equal(a, b)` (lines 92-165)** — 13 branches plus
   a Block branch (lines 154-160) and a structural-hash
   fallback (lines 164-165). The fallback returns
   `structural_hash(a) == structural_hash(b)` for nodes the
   enumerated branches don't cover. **Is this a silent
   second-stage fallback?**

   The fallback is documented at lines 161-164 as
   "Conservative default: fall back to hash equality.
   Reaching this branch means _ast_equal hit a node type the
   explicit enumeration doesn't cover; SHA-256 is collision-
   resistant so trusting the hash is a safe (and very rare)
   fallback." This is the EXPLICIT JUSTIFICATION for the
   fallback — it's defensible because:
   - SHA-256 collision probability is cosmologically small.
   - Reaching the fallback means we already hash-matched at
     line 328-329, so the second-stage check is redundantly
     checking the same hash.
   - The first-stage match at line 328 is already SHA-256
     hash equality; the fallback at line 164 is the same
     check restated. So in the fallback path, the two
     hashes are guaranteed to be equal (by construction —
     we wouldn't be in `_ast_equal` unless `_maybe_share`
     already saw a hash match).

   Wait — that's actually concerning. If the first-stage
   hash matches but the explicit `_ast_equal` branches all
   return False (e.g., `type(a) is not type(b)` at line 99),
   we never reach the fallback. The fallback is only reached
   when `_ast_equal` falls through every enumerated branch
   without matching, which means the node type is one of
   the cases NOT covered (e.g., a future expression node
   added to `_SHAREABLE` but not added to `_ast_equal`).
   In that case, returning `True` based on hash-only is
   "trusting SHA-256 to disambiguate". The risk is a
   collision between two structurally-distinct nodes of an
   unenumerated type. Per the docstring, this is acceptable
   in Phase-0.

   **But is this a build-time hazard?** Yes — if `_SHAREABLE`
   gains a new entry without `_ast_equal` getting a
   corresponding branch, the fallback ALWAYS fires for that
   type, defeating the collision-defense purpose. **Static-
   check-at-authoring-time hazard**: any new entry in
   `_SHAREABLE` must be paired with a new `_ast_equal`
   branch. This invariant is enforced only by code review;
   no static check.

   Filed as a future-tracking item in the "Deferred
   observations" section below. **Not a silent-failures
   finding for cycle 18** because:
   - The current `_SHAREABLE` (13 entries) is fully
     covered by `_ast_equal` (12 enumerated branches +
     1 Block branch). I verified this by mapping each
     `_SHAREABLE` entry to its corresponding `_ast_equal`
     branch: IntLit→101, FloatLit→103, BoolLit→105,
     StrLit→107, CharLit→109, Name→111, Unary→113,
     Binary→115, Call→119, If→125, Cast→129,
     TupleLit→132, ArrayLit→136, Field→140, Index→142,
     Range→148. All 16 shareable types are covered.
     (Block at line 154 is for descend-through, not for
     `_SHAREABLE`-membership.)
   - The fallback path is unreachable today with the
     current type enumeration.
5. **`_stmt_equal(a, b)` (lines 168-182)** — handles Let,
   ConstStmt, ExprStmt with explicit branches; falls back
   to `structural_hash(a) == structural_hash(b)` for
   unenumerated statement types. Same fallback rationale
   as `_ast_equal`. Not a silent-failure window.
6. **`_Sharer.__init__` (lines 186-190)** — initializes
   `self.merged = 0` and `self._canon: dict[str, Any] =
   {}`. No silent surface.
7. **`_Sharer.share_in(node)` (lines 192-259)** — descends
   into a node. Branch coverage:
   - `node is None: return None` (line 196-197) — safe
     no-op, documented.
   - `Block` (lines 201-206) — walks `stmts` and
     `final_expr`; doesn't share the Block itself.
   - `Match` / `For` / `While` / `Loop` (lines 210-236) —
     walks but doesn't share (scope/control-flow
     dependence). Each branch reads `node.X = self.
     _maybe_share(node.X)` or `self.share_in(node.body)`.
   - `_SHAREABLE` (lines 244-245) — routes to
     `_maybe_share`.
   - Conservative walker (lines 250-258) — for unknown
     node types, walks attribute name list `(inner,
     target, transformation, verifier, value, operand,
     expr)` using `getattr(node, attr, None)`. The
     `if v is None or isinstance(v, (str, int, float,
     bool)): continue` guard skips non-AST values. The
     `if hasattr(v, "span")` check at line 257 guards
     against accidentally recursing into non-AST values
     (like type nodes).

   **Is the conservative walker a silent-failure window?**
   It silently descends into all known attribute names and
   silently skips unknown attribute names. If a future AST
   node added a new child-attr name (e.g., `payload`), it
   wouldn't be walked. **But the consequence**: a missed
   sharing opportunity, not a miscompile. The walker
   doesn't REPLACE the node with a wrong value; it just
   doesn't visit some children. The node graph remains
   structurally correct.

   **Static-check-at-authoring-time hazard**: any new
   non-`_SHAREABLE` AST node type with child attrs not in
   the hand-curated list would silently skip those
   children during hash-cons. Filed as a future-tracking
   item; not a silent-failures finding for cycle 18
   because no such future name exists in the current tree.
8. **`_Sharer._share_stmt(stmt)` (lines 264-282)** —
   handles Let, ConstStmt, ExprStmt with explicit
   branches, then walks an attribute name list `("expr",
   "value", "target")` for other statement kinds. Same
   conservative-walker pattern as `share_in`. Same
   static-check-at-authoring-time hazard reasoning. Not a
   silent-failure window.
9. **`_Sharer._maybe_share(node)` (lines 287-342)** —
   the hash-cons heart. Branch coverage:
   - `node is None: return None` (line 291-292) — safe.
   - Non-`_SHAREABLE` (lines 293-295) — routes to
     `share_in`.
   - Bottom-up child sharing (lines 299-325) — covers
     Unary, Binary, Call, If, Cast, TupleLit, ArrayLit,
     Field, Index, Range explicitly. Note: the
     `IntLit`/`FloatLit`/`BoolLit`/`StrLit`/`CharLit`/
     `Name` types are leaf-only and have no children to
     share, so they correctly aren't enumerated in this
     branch list. Verified by inspection: each leaf type
     is in `_SHAREABLE` (passes the type check at 293)
     but has no child-walking branch (correctly, since
     they have no children).
   - Hash lookup (line 328): `h = structural_hash(node)`.
   - Canon lookup (line 329): `canon = self._canon.get(h)`.
     `dict.get` defaults to None.
   - New canon registration (lines 330-332): if no canon,
     register `self._canon[h] = node` and return node.
   - Collision check (lines 333-339): if canon exists,
     check `_ast_equal(canon, node)`. If they differ,
     raise `HashConsError` LOUDLY (trap-20001).
   - Share (lines 340-342): increment `self.merged` and
     return canon.

   **Is the collision check a silent-failure window?**
   The line 334 `if not _ast_equal(canon, node): raise
   HashConsError(...)` is LOUD by construction — the user
   sees the trap-20001 message. The HashConsError is
   caught at `check.py:284-318` outer-except chain and
   surfaces as an internal-error/compiler-bug message
   with rc=1.

   **Is the fallback path inside `_ast_equal` (line 165)
   silent?** Per the walk above, the fallback returns
   `structural_hash(a) == structural_hash(b)`. If both
   hashes match, the fallback returns True. If they
   differ, False — which would cause the collision check
   at line 334 to raise (because line 329 already pulled
   `canon` by hash equality with `node`, so the hashes
   ALREADY matched). So the fallback inside `_ast_equal`
   always returns True when reached from this path.
   **Is "always-True from this path" a silent miscompile?**
   It means a hash-only fallback for an unenumerated node
   type would silently merge two structurally-distinct
   nodes that hash-collided. This is the documented
   Phase-0 risk per the module docstring at lines
   43-44: "Trap-id 20001: hash collision detected. The
   structural hasher uses SHA-256, which is collision-
   resistant — but we still cheap-check post-share
   equivalence via `_ast_equal`, and raise on any false
   positive so a future hash regression surfaces loudly."

   But the trap-20001 only fires when `_ast_equal`
   returns False; if `_ast_equal` falls through to the
   hash-only fallback and returns True, the trap doesn't
   fire. The collision defense is therefore **partial**:
   it catches collisions for the 16 enumerated types but
   accepts collisions for any future un-enumerated type.

   **Severity assessment**: SHA-256 collision is
   cosmologically small; an un-enumerated `_SHAREABLE`
   type today would still benefit from SHA-256's
   pre-image resistance. The risk is theoretical, not
   exploitable. The hazard is **static-check-at-
   authoring-time**: any new `_SHAREABLE` entry MUST be
   paired with a new `_ast_equal` branch to maintain the
   trap-20001 defense. This invariant is enforced only
   by code review.

   Filed as a future-tracking item; not a silent-failures
   finding for cycle 18 because all current `_SHAREABLE`
   entries are covered.

**Conclusion**: `hash_cons.py` interior has zero exception
handlers (matching the cycle-13 raise-only inventory result
at line 335), and every `.get(...)`-style fallback is either
documented Phase-0 design (SHA-256 hash-only fallback in
`_ast_equal`/`_stmt_equal`) or static-check-at-authoring-time
hazard (any new `_SHAREABLE` entry needs `_ast_equal`
branch + child-walker branch). All current types are fully
covered; no live silent-failure window today. Confirmed
non-finding for cycle 18.

### Doc-only commit at HEAD (0243d5c)

The diff is entirely in `docs/APPROACH_A_PLAN.md` and
`docs/helix-pre-phase-A-finalization-research.md`. Markdown
files cannot introduce a runtime silent-failure window. The
plan document references hazards that the cycle-18 audit
should be aware of:
- "Stage 28.8.1 (NEW): codegen determinism harden" — flagged
  9 `id(op):x` call sites in helixc/backend/x86_64.py that
  leak Python object memory addresses into symbol names.
  This is a **determinism** concern, not a silent-failure
  concern (the symbols ARE emitted, just not byte-identical
  across runs). Not a silent-failures audit finding.
- "Stage 28.8.2: shared AST walker library" — would extract
  the attribute-list dispatch pattern from panic_pass /
  unsafe_pass / deprecated_pass / grad_pass / struct_mono.
  This pattern is what powers the conservative-walker
  branches in `_Sharer.share_in` and `_Sharer._share_stmt`
  audited above. The future extraction may consolidate the
  hand-curated attribute name lists into one canonical
  source, reducing the static-check-at-authoring-time
  hazard. Filed as a future-tracking item; not a
  silent-failures finding.

Confirmed non-finding for the doc-only commit.

### Global `except: pass` hunt

`grep -nE 'except\s*:\s*pass|except\s+Exception\s*:\s*pass|
except\s+BaseException' helixc/` returns ONE match:

- `helixc/frontend/autodiff.py:998` —
  `# this 'except Exception: pass' swallowed every error in`

This is a **comment** in autodiff.py describing the prior
(now-fixed) silent-failure pattern. The actual code at line
1012 is `except (OverflowError, ZeroDivisionError, ValueError,
TypeError)` — a narrow typed except.

**Zero genuine `except: pass` patterns in production code.**
Stable non-finding.

---

## Carryover findings status (cycles 1-17) — unchanged

The cycle-18 re-audit closed nothing (read-only by design)
and introduced no new finding. The carryover ledger is
identical to cycle 17's closing snapshot.

| Carryover | Severity | Cycle-18 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — not addressed. Highest-priority unaddressed-CRITICAL. |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| C5-10 (lower_ast.py:2113-2117 + 2079-2092 + 2093-2101 + :280-283 + :2064-2068) | LOW | **still open** — not addressed; not re-flagged per the strict re-flag rule |
| monomorphize_safe docstring drift | (housekeeping) | **still open** |
| D-vs-Quote diagnostic text | (housekeeping) | **still open** |
| C7-1 test-coverage gap | (housekeeping) | **still open** |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | CLOSED by cycle 9 |
| C8-2 (cycle-8 LOW) | LOW | CLOSED by cycle 9 |
| C9-1 (cycle-9 LOW) | LOW | CLOSED by cycle 10 |
| C13-1 (cycle-13 HIGH, DCE drops TRACE_EXIT operand) | HIGH | CLOSED by cycle 14 fix-sweep at 1e4c3e6 |
| C16-1 (cycle-16 HIGH, wide-array-elem silent trunc) | HIGH | CLOSED by cycle 17 fix-sweep at c6136d4 |

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

(none)

---

## MEDIUM FINDINGS

(none)

---

## LOW FINDINGS

(none)

---

## Re-audit verification on 0243d5c (production surface identical to cycles 16 + 17)

| Re-audit pass | C13 | C14 | C15 | C16 | C17 | C18 | Stability |
|---|---|---|---|---|---|---|---|
| `_emit_env_error` strip helper (check.py:246-255) | clean | clean | clean | clean | clean | clean | stable |
| Outer-except topology (check.py:284-318) | clean | clean | clean | clean | clean | clean | stable |
| Finally drain-failure suppressor (check.py:319-337) | clean | clean | clean | clean | clean | clean | stable |
| Backend-call wraps (check.py:618,649,663) | clean | clean | clean | clean | clean | clean | stable |
| AD-warning narrowed excepts (autodiff.py:155,1012) | clean | clean | clean | clean | clean | clean | stable |
| const_fold defensive folds (const_fold.py:250,324,349,401) | clean | clean | clean | clean | clean | clean | stable |
| Quote-handle fallback (lower_ast.py:2115) | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| diagnostics isatty fallback (diagnostics.py:76) | non-finding | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| `getattr(it, "is_kernel", False)` (check.py:641) | non-finding | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| lower_ast.py try/finally scope at :596, :1800 | clean | clean | clean | clean | clean | clean | stable |
| backend/x86_64.py attrs.get defaults | clean | clean | clean | clean | clean | clean | stable |
| backend/ptx.py, elf_dyn.py zero-except | clean | clean | clean | clean | clean | clean | stable |
| frontend/parser.py:375 ValueError -> ParseError re-raise | clean | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:415,423 TypeError_ -> diag append | clean | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:636 ValueError -> Optional None | clean | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:203 ValueError -> return expr | clean | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:759 ShapeFoldError -> diag list | clean | clean | clean | clean | clean | clean | stable |
| frontend/grad_pass.py:639-643 frozen-dataclass cache fallback | C13 fresh: clean | clean | clean | clean | clean | clean | stable |
| frontend/pytree.py:293-296 validate_pytree diagnostic collection | C13 fresh: clean | clean | clean | clean | clean | clean | stable |
| frontend/hash_cons.py:335 raise HashConsError | C13 fresh: clean | clean | clean | clean | clean | clean | stable |
| frontend/flatten_impls.py:88 raise DuplicateMethodError | C13 fresh: clean | clean | clean | clean | clean | clean | stable |
| frontend/flatten_modules.py:67,77 raise FlattenError | C13 fresh: clean | clean | clean | clean | clean | clean | stable |
| frontend/trace_pass.py:67 raise OverflowError | C13 fresh: clean | clean | clean | clean | clean | clean | stable |
| ir/passes/effect_check.py:228 raise EffectError | C13 fresh: clean | clean | clean | clean | clean | clean | stable |
| dce.py SIDE_EFFECT_KINDS frozenset (incl. C14 +TRACE_ENTRY/EXIT) | (n/e) | C14 fresh: clean | clean | clean | clean | clean | stable |
| cse.py PURE_KINDS dual-check vs SIDE_EFFECT_KINDS | (n/e) | C14 fresh: clean | clean | clean | clean | clean | stable |
| fdce.py call-graph source check vs TRACE_* | (n/e) | C14 fresh: clean | clean | clean | clean | clean | stable |
| x86_64.py TRACE_EXIT operand consumer guard | (n/e) | C14 fresh: clean | clean | clean | clean | clean | stable |
| lower_ast.py synthesized-const sentinel (line 573-574, 1891-1892) | (n/e) | C14 fresh: clean | clean | clean | clean | clean | stable |
| lexer.py:399-402 `\u` escape ValueError -> LexError re-raise | (n/e) | (n/e) | C15 fresh: clean | clean | clean | clean | stable |
| lower_ast.py:280-283 flat-path index ValueError -> None (C5-10 Pat C) | (n/e) | (n/e) | C15 fresh: C5-10 carryover | stable carryover | stable carryover | stable carryover | stable carryover |
| lower_ast.py:2064-2068 Field-of-Field flat-path ValueError -> -1 (C5-10 Pat C) | (n/e) | (n/e) | C15 fresh: C5-10 carryover | stable carryover | stable carryover | stable carryover | stable carryover |
| struct_mono.py:445-456 ShapeFoldError + ValueError -> diags | (n/e) | (n/e) | C15 fresh: clean | clean | clean | clean | stable |
| backend/x86_64.py raise-only inventory (24 sites) | (n/e) | (n/e) | C15 fresh: clean | clean | clean | clean | stable |
| cse.py + fdce.py zero try/except/raise | (n/e) | (n/e) | C15 fresh: clean | clean | clean | clean | stable |
| effect_check.py full-module audit | (n/e) | (n/e) | (n/e) | C16 fresh: clean | clean | clean | stable |
| totality.py full-module audit | (n/e) | (n/e) | (n/e) | C16 fresh: clean | clean | clean | stable |
| cse.py `_find_value_by_id` dead helper (line 122-134) | (n/e) | (n/e) | (n/e) | C16 fresh: clean | clean | clean | stable |
| x86_64.py:983-1003 `_check_array_elem_size_supported` helper (C16-1 fix) | (n/e) | (n/e) | (n/e) | (n/e) | C17 fresh: clean | clean | stable |
| x86_64.py:2743 LOAD_ELEM C16-1 guard wiring | (n/e) | (n/e) | (n/e) | (n/e) | C17 fresh: clean | clean | stable |
| x86_64.py:2764 STORE_ELEM C16-1 guard wiring | (n/e) | (n/e) | (n/e) | (n/e) | C17 fresh: clean | clean | stable |
| test_codegen.py:437-475 `test_c16_1_wide_array_elem_traps_at_codegen` | (n/e) | (n/e) | (n/e) | (n/e) | C17 fresh: clean | clean | stable |
| PTX + dyn-ELF backends carry no LOAD_ELEM/STORE_ELEM today | (n/e) | (n/e) | (n/e) | (n/e) | C17 fresh: clean | clean | stable |
| **fdce.py full-module audit (entry_fn short-circuit, CALL/MODIFY/QUOTE callee scrape, roots, fixpoint, dead-fn deletion)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C18 fresh: clean** (zero except/raise/try; 4 `.get` defaults each justified by upstream-producer-guarantees or downstream-LOUD-link-error; the entry_fn short-circuit at lines 30-31 is the documented "don't silently empty the module" guard; LOUD downstream via Buffer.patch unresolved-symbol raise) | new |
| **hash_cons.py interior walk (lines 82-342: hash_cons entry, _ast_equal/_stmt_equal, _Sharer.share_in/_share_stmt/_maybe_share)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C18 fresh: clean** (single raise at line 335 — already cycle-13-clean; SHA-256 hash-only fallback at lines 164-165 documented Phase-0 design; all 16 `_SHAREABLE` entries covered by `_ast_equal` branches; collision defense LOUD via trap-20001 / HashConsError; future hazard: any new `_SHAREABLE` entry must add `_ast_equal` branch — static-check-at-authoring-time, not runtime silent) | new |
| **Doc-only commit 0243d5c (Phase A staging refresh)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C18 fresh: clean** (markdown only; cannot introduce runtime silent failure; references future stages 28.8.1 / 28.8.2 as deterministic + walker-lib refactors — both reduce risk going forward) | new |
| Global `except: pass` hunt (zero matches in production) | clean | clean | clean | clean | clean | clean | stable |

### Specific cycle-18 items re-checked clean

- **No new production commits since cycle 17** → production-
  code surface identical to cycles 16 + 17: `git diff
  c6136d4..HEAD -- 'helixc/' '*.py' '*.hx'` returns empty.
  By construction the cycle-17 clean verdict propagates to
  cycle 18 for the silent-failures production-code lens.
- **fdce.py full-module audit**: zero try/except/raise, four
  `.get` calls each justified. The entry_fn short-circuit
  is the documented anti-silent-empty-module guard.
  Downstream link-failure is LOUD via Buffer.patch
  unresolved-symbol raise. Confirmed non-finding.
- **hash_cons.py interior**: the trap-20001 collision raise
  is LOUD by construction; the SHA-256 hash-only fallback in
  `_ast_equal`/`_stmt_equal` is documented Phase-0 design
  and the fallback path is unreachable for the current
  `_SHAREABLE` enumeration (verified by exhaustive
  mapping). The `_Sharer.share_in` conservative walker
  (lines 250-258) silently skips non-AST and unknown-attr
  values; missed sharing is a missed optimization, not a
  miscompile. Confirmed non-finding.
- **hash_cons + check.py-without-flatten_modules
  observation**: noted but NOT a new finding. Filed under
  Deferred observations because the downstream failure
  mode is LOUD (`unresolved symbol: foo::bar` at
  Buffer.patch) and the concern is the same family as
  carryover audit-C4-8 (pipeline-stage omission). The
  silent-failures lens scope is satisfied by the LOUD
  downstream error path.
- **Doc-only commit at HEAD (0243d5c)**: cannot introduce
  runtime silent failure. The referenced future stages
  (28.8.1 codegen determinism, 28.8.2 shared AST walker)
  both reduce future silent-failure risk by consolidating
  hand-curated attribute name lists. Filed as
  future-tracking items.
- **Global `except: pass` hunt**: only one grep match in
  production, which is a COMMENT in autodiff.py:998
  describing the prior (now-fixed) bare-except pattern.
  Zero genuine `except: pass` arms in production code.
  Stable non-finding.

### Cross-stage interactions re-checked (cycle 18)

- **fdce CALL/QUOTE callee-scrape → backend link-fixup**:
  fdce's `.get` defaults on missing target/verifier_fn/
  ast_pretty don't hide failures because the downstream
  Buffer.patch (line 120-121) raises `ValueError(
  "unresolved symbol: ...")` LOUDLY on link-time. The
  check.py outer-except chain (lines 284-318) catches
  this and emits user-visible "internal error" with rc=1.
  Not silent.
- **hash_cons `HashConsError` → check.py outer-except**:
  the raise at line 335 propagates through the
  typecheck → IR → backend chain to check.py's broad-
  Exception arm. User sees the trap-20001 message
  verbatim. Not silent.
- **No new production commits since cycle 17**: the
  cycle-17 verdict propagates. Confirmed.

### Did the cycle-18 fresh-eyes rotation surface any overlooked silent-failure window?

Two production modules audited at full line-by-line
resolution (`fdce.py` 91-line module covered exhaustively;
`hash_cons.py` interior lines 82-342). Each has zero new
silent-failure window beyond what is already in the
carryover ledger. The doc-only commit at HEAD cannot
introduce a runtime silent failure.

**Conclusion**: zero new silent-failure findings for cycle 18.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-19 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 18 did not address (read-only
  re-audit). **STILL THE HIGHEST-PRIORITY ITEM** for any
  future fix-sweep — the only remaining CRITICAL across the
  audit series. As the clean-counter advances toward the
  5/5 Stage-29 gate, the question of whether the gate
  requires CRITICAL=0-open (stricter) or merely
  5-consecutive-clean (lenient) becomes load-bearing.
- **Carryover audit-C4-4 (D9 paper-only)**: still open
  HIGH. Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks (Patterns A, B, C —
  including the cycle-15-enumerated :280-283 and :2064-2068
  sites)**: still open LOW. Not addressed; not re-flagged.
- **monomorphize_safe docstring drift**: still open
  (cycle-6 deferred).
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred).
- **C7-1 test-coverage gap**: still open. Cycle 18 also did
  not add the 4 `_compatible(TyMemTier, TyVar)` regression
  tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix
  edge cases**: still no callee triggers either. Not
  findings.
- **TRACE_EXIT operand-less defensive guard
  (x86_64.py:2495)**: the `if op.operands:` guard tolerates
  a hypothetical operand-less TRACE_EXIT. Future-tracking
  item if the trace machinery evolves. Not a finding for
  cycle 18.
- **cse.py `_find_value_by_id` dead helper**: zero callers.
  Code-hygiene candidate for a future code-review lens
  pass, NOT a silent-failures finding.
- **OP_EFFECTS completeness (effect_check.py:40-49)**: any
  new effect-bearing op kind MUST be added to OP_EFFECTS by
  the author at IR-design time. Static-check-at-authoring-
  time hazard. Filed as a future-tracking item; not a
  silent-failures finding for cycle 18.
- **totality.py `_children` attribute coverage**: the
  hand-curated attribute name list (lines 89-92) is
  exhaustive for the current AST shape. Static-check-at-
  authoring-time hazard. Filed as a future-tracking item;
  not a silent-failures finding for cycle 18.
- **C16-1 fix's `wide_widths` set completeness
  (x86_64.py:983-1003)**: any new wide-scalar name added to
  `TIRScalar` (currently `name: str` accepts any string)
  MUST also be added to `wide_widths` or LOAD_ELEM /
  STORE_ELEM would silently 32-bit-truncate it. Static-
  check-at-authoring-time hazard. No new wide-scalar name
  exists in the current tree. Future-tracking item filed
  in cycle 17, unchanged in cycle 18.
- **Regression test `hard` dead-bind
  (test_codegen.py:461)**: noted cycle 17. Code-review-lens
  observation, NOT a silent-failures finding.
- **`hash_cons.py` `_SHAREABLE` vs `_ast_equal` paired-
  invariant** (NEW cycle-18 future-tracker): any new entry
  in `_SHAREABLE` (lines 68-79) MUST be paired with a
  corresponding branch in `_ast_equal` (lines 92-165) and
  with a child-walking branch in `_Sharer._maybe_share`
  (lines 299-325). Currently all 16 `_SHAREABLE` entries
  are covered. Static-check-at-authoring-time hazard. The
  Stage 28.8.2 shared-AST-walker library (planned in
  0243d5c) could consolidate this invariant into one
  canonical source. Filed as a future-tracking item.
- **`hash_cons.py` `_Sharer.share_in` conservative-walker
  attribute name list (NEW cycle-18 future-tracker)**: the
  hand-curated list at line 250 (`"inner", "target",
  "transformation", "verifier", "value", "operand",
  "expr"`) and at line 278 (`"expr", "value", "target"`)
  is exhaustive for the current AST. Static-check-at-
  authoring-time hazard. Same Stage-28.8.2 consolidation
  applies.
- **check.py-without-flatten_modules (NEW cycle-18
  observation, FILED NOT-FLAGGED)**: `helixc/check.py:565`
  calls `lower(prog)` after invoking flatten_impls but
  WITHOUT calling `flatten_modules` first. A user program
  with `mod foo { fn bar() }` + `main() { foo::bar() }`
  would:
  1. Parse → prog.items contains ModBlock(name="foo",
     items=[FnDecl("bar")]).
  2. typecheck → silently doesn't descend into ModBlock
     (line 404-426 only iterates FnDecl/StructDecl/
     EnumDecl); Path-call `foo::bar(x)` returns
     `TyUnknown` via typecheck.py:1264.
  3. flatten_impls → no-op (no ImplBlock).
  4. (no flatten_modules → ModBlock items still present)
  5. lower(prog) → silently skips ModBlock items (line
     181-186 only walks FnDecl); the IR module is
     missing the `bar` function.
  6. backend codegen → CALL op with target="foo::bar"
     becomes a fixup.
  7. Buffer.patch() at link time → raises
     `ValueError("unresolved symbol: foo::bar")` LOUDLY.
  8. check.py outer-except catches the ValueError and
     emits "internal error: unresolved symbol: foo::bar"
     with rc=1.

  This is the same shape as carryover **audit-C4-8**
  (check.py pipeline-stage omission, downstream-LOUD).
  The downstream failure mode IS LOUD, so this is NOT a
  silent-failures finding. **The UX could be improved**
  (the user sees "internal error" but they wrote valid
  syntax) — that's a **code-review-lens** concern about
  diagnostic quality. Filed for code-review-lens
  consideration, not silent-failures-lens. Not
  re-flagged here per the strict re-flag rule
  (structurally the same as audit-C4-8).

---

## Cycle 17 vs cycle 18 — clean-cycle counter check

Cycle 17 was the 1st clean of the re-accumulated post-
C16-1-fix window (counter 1/5). The user directive for
cycle 18 explicitly instructs: re-audit with rotated
spot-check surface, do not re-flag prior-cycle carryovers
unchanged since cycle 17.

The cycle-18 re-audit honors that directive:
- `audit-C4-1 CRITICAL`, `audit-C4-4 HIGH`, `audit-C4-8
  LOW`: not re-flagged.
- `C5-10 LOW`: not re-flagged.
- `monomorphize_safe docstring drift`, `D-vs-Quote
  diagnostic text`, `C7-1 test-coverage gap`: not
  re-flagged.

Cycle 18 produces **zero NEW findings of any severity**,
so the clean-cycle counter advances to **2/5** under the
strict criterion — subject to the parallel type-design +
code-review audit lenses also being clean for cycle 18.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 18 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW).**

---

## Cycle 18 status

**Cycle 18 IS CLEAN** for the silent-failure audit lens. Per
the strict criterion (zero findings of ANY severity), the
0-finding result satisfies the clean-cycle gate for this
audit lens.

### Stop-the-line determination: **NO**

Cycle 18 is clean — no stop required for this lens.

### Cycle 18 -> NEW FINDINGS COUNT for the strict-clean gate: 0

(0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter
advances to **2/5** for this audit lens (cycle 17 silent-
failures clean = 1/5; cycle 18 = 2/5 of the re-accumulated
window).

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 8: 2 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW).
- Cycle 9: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 10: 0 findings.
- Cycle 11: 0 findings.
- Cycle 12: 0 findings.
- Cycle 13: 0 findings (silent-failures lens; code-review
  lens found C13-1 HIGH, addressed by cycle-14 fix-sweep).
- Cycle 14: 0 findings (silent-failures lens).
- Cycle 15: 0 findings (silent-failures lens).
- Cycle 16: 0 findings (silent-failures lens; type-design
  lens found C16-1 HIGH, addressed by cycle-17 fix-sweep).
- Cycle 17: 0 findings (silent-failures lens).
- Cycle 18: 0 findings (silent-failures lens). <- here

Trend: **9 consecutive clean cycles** on the silent-failures
lens (10 through 18). The global strict-clean counter is
2/5 because cycle 16's type-design lens broke the prior
3-clean-cycle accumulation, resetting the global counter to
0; cycle 17 + cycle 18 are the first two cycles of the
re-accumulated window.

### Estimated remaining open findings going into cycle 19

- Cycle 1: 13 new (all fixed -> 0 open).
- Cycle 2: 6 new (all fixed -> 0 open).
- Cycle 3: 6 new (all fixed -> 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 5 silent-failure: 4 new — 3 closed by cycle 6.
  1 still open (C5-10 LOW).
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new — both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10-18 silent-failure: 0 new each. <- here
- Cycle 13 code-review: C13-1 HIGH — CLOSED by cycle 14
  fix-sweep.
- Cycle 16 type-design: C16-1 HIGH — CLOSED by cycle 17
  fix-sweep at c6136d4.
- Prior audits (stage 5-6 + 7-8 + 9-18): ~20 still-open
  (unchanged going into cycle 19).
- Cycle 18 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-18 new) = **>=23 open findings** going into
  cycle 19. (Net 0 delta vs cycle 17's silent-failure
  tally.)

Recommend prioritizing in this order for the cycle-19 fix
batch (if user elects to land fixes between clean re-
audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks).
4. **C7-1 test-coverage gap**.
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).
7. **check.py-without-flatten_modules diagnostic UX**
   (NEW cycle-18 deferred — code-review-lens, not
   silent-failures-lens; would replace
   `internal error: unresolved symbol: foo::bar` with
   a clearer up-front "block modules are not yet
   supported in check.py emit pipeline" diagnostic).

The "5 clean cycles before Phase 0 deprecation" goal
requires the strict criterion (zero findings of any
severity, all three lenses) to be met for 5 CONSECUTIVE
cycles. Cycle 17 = 1/5; cycle 18 = 2/5 of the
re-accumulated window. Three more clean cycles (19, 20,
21) needed across all three lenses to fire the gate
(assuming parallel type-design + code-review lenses
remain clean).

**Cycle 18 status: CLEAN**
**Counter status: 2/5** (cycles 17 + 18 silent-failures
both clean; subject to parallel type-design + code-review
lenses also being clean for cycle 18).
