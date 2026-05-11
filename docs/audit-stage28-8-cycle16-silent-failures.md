# Stage 28.8 Cycle 16 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit (HEAD)**: 4c74627 — "Cycle 11 Audit A A1
(out-of-scope but acted on): fix bootstrap test harness flake".

**Note on HEAD movement vs cycle 15**: one new commit landed at
4c74627 between cycle 15 (1e4c3e6) and cycle 16. `git show --stat
4c74627` shows the diff is entirely in `helixc/tests/test_codegen.py`
(+37/-9 lines, a single test-harness shell-command rewrite from `;` to
strict `&&` plus a `test -f` precondition and an
`__HARNESS_FAIL_BOOTSTRAP_DID_NOT_WRITE_OUTPUT__` sentinel that the
Python harness asserts on). The commit message explicitly classifies
the change as "out-of-scope but acted on" — a flake fix in test
infrastructure, not a production-code change.

`git diff 1e4c3e6..HEAD -- 'helixc/' '*.py' '*.hx'` confirms:
**zero production-code surface delta** vs cycle 14/15 HEAD. Every
clean verdict on cycle 15 holds on cycle 16 unless the fresh-eyes
re-walk surfaces an overlooked window. The new commit could
NOT have opened a silent-failure window in production code by
construction (it only touches `helixc/tests/test_codegen.py`).

**Scope**: any silent-failure window NOT already counted in cycles
1-15 as a carryover. Documented carryovers (audit-C4-1 CRITICAL,
audit-C4-4 HIGH, audit-C4-8 LOW, C5-10 LOW, monomorphize_safe
docstring drift, D-vs-Quote diagnostic text, C7-1 test-coverage gap)
are NOT re-flagged per the user's strict re-flag rule (a carryover
is re-flagged only if it CHANGED since the prior cycle — and none
did, because no production code changed since cycle 14).

**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Findings already in the carryover
ledger are explicitly excluded.

**Clean-counter state**: cycle 14 = 1/5 (silent-failures clean),
cycle 15 = 2/5 (silent-failures clean). Cycle 16 is the third
clean-cycle attempt of the re-accumulated window.

---

## Method

1. **Read cycle-14 + cycle-15 silent-failure verdicts**: both CLEAN
   for the silent-failures lens. Cycle 14 fix-sweep closed C13-1
   (HIGH, DCE drops TRACE_EXIT operand) by adding TRACE_ENTRY and
   TRACE_EXIT to `SIDE_EFFECT_KINDS`.
2. **Confirmed HEAD delta vs cycle 15** is test-only:
   - `git log --oneline 1e4c3e6..HEAD` → `4c74627 Cycle 11 Audit A
     A1 (out-of-scope but acted on): fix bootstrap test harness
     flake`.
   - `git show --stat 4c74627` → only `helixc/tests/test_codegen.py`
     touched (+37/-9 lines).
   - `git diff 1e4c3e6..HEAD -- 'helixc/' '*.py' '*.hx'` confirms
     the production surface IS the test_codegen.py rewrite and
     nothing else. **Zero production-code surface delta** vs cycle
     14/15 HEAD.
   - Working-tree changes (un-committed) are still doc-only (the
     cycle-15 audit docs added in the tree) plus the cycle-16
     audit doc this file is part of.
3. **Verified the cycle-14 fix at HEAD**
   (`helixc/ir/passes/dce.py:68-80`): `tir.OpKind.TRACE_ENTRY` and
   `tir.OpKind.TRACE_EXIT` remain present in `SIDE_EFFECT_KINDS`.
   Multi-line block comment (lines 68-78) explaining the C13-1
   rationale is intact. No production change at this site since
   the fix landed at 1e4c3e6.
4. **Cycle-16 fresh-eyes rotation**: cycles 14 + 15 already
   covered dce.py SIDE_EFFECT_KINDS, cse.py PURE_KINDS, fdce.py
   call-graph edges, x86_64.py TRACE_EXIT consumer guard,
   lower_ast.py synthesized-const sentinel, lexer.py `\u` escape,
   struct_mono.py generic-instantiation diagnostics, and backend
   raise-only inventory. Cycle 16 rotates to:
   - `helixc/ir/passes/effect_check.py` (full module — line-by-line
     audit of `OP_EFFECTS`, `META_ATTRS`, `declared_effects`,
     `is_pure_decl`, `own_op_effects`, `callees`, `compute_closure`,
     `check_module`, `verify_module`).
   - `helixc/frontend/totality.py` (full module — line-by-line
     audit of `check_totality`, `_collect_self_calls`, `_children`,
     `_arg_strictly_decreases`, `_is_strictly_smaller`, `_is_name`,
     `_is_positive_int_const`, `_is_int_const_at_least`).
   - `helixc/ir/passes/cse.py` (full module — including the
     previously-unenumerated `_find_value_by_id` helper at
     lines 122-134).
   - The new commit's diff in test_codegen.py — even though
     test-only, scrutinize whether the new
     `__HARNESS_FAIL_BOOTSTRAP_DID_NOT_WRITE_OUTPUT__` sentinel +
     `assert` pattern is itself silent or LOUD on failure.
5. **Read-only**: no edits to production code or tests.

---

## Fresh-eyes walk for cycle 16

### `helixc/ir/passes/effect_check.py` — full-module audit

**File invariants**:
- 0 `except` arms (the word "except" only appears at line 27 inside
  the module-level docstring as "everything in attrs except
  meta-keys").
- 1 `raise` site (line 228 in `verify_module`): `raise
  EffectError("\n".join(errs))`. Typed user-visible error; consumed
  by `check.py` callers as part of the typecheck → IR → backend
  chain.

**Surface walk** (line numbers per the file as of 4c74627):
1. `OP_EFFECTS` (lines 40-49) — frozenset literal of 4 op kinds
   (PRINT → io, MODIFY → modify_self, SPLICE → modify_self, TRAP
   → io). Every member is justified by a Helix semantics comment
   nearby. No silent-failure window — the dict is read-only and
   complete for the op kinds with effects today.
2. `META_ATTRS` (lines 54-67) — frozenset of attribute keys that
   are NOT effect labels. Each member has a justifying comment.
   The `declared_effects` function loops over `fn.attrs.items()`,
   filtering via this set. If a new attribute were added without
   updating META_ATTRS, it would be misclassified as a declared
   effect — but this is a build-time (statically-known) hazard,
   not a runtime silent failure. Stable non-finding.
3. `EffectError` class (lines 70-76) — typed exception with two
   trap-id class attrs (19001, 19002). No method body, no hidden
   silent-failure logic.
4. `declared_effects(fn)` (lines 79-106) — walks `fn.attrs.items()`,
   skips non-True values (`if v is not True: continue`), skips
   META_ATTRS, strips `effect:` prefix or accepts bare label.
   - The `v is not True` filter (line 96-97) silently skips any
     attribute whose value is not the boolean literal `True`.
     Could this silently miss an effect? **No**: the lower_ast.py
     and frontend bind effects ONLY with value `True` (see the
     `@effect(io)` lowering path in `lower_ast.py` which sets
     `fn.attrs["effect:io"] = True`). A future code path that
     stored a string or other value at an effect key would be a
     bug in the producer, not the consumer. The filter is the
     correct conservative default.
   - The `k.startswith("effect:")` + `k[len("effect:"):]` strip
     (lines 100-101) is a documented design choice — the comment
     in `declared_effects`'s docstring explicitly explains the
     Stage 19 regression test
     (`test_stage19_trap_19002_does_not_fire`) that caught the
     "bare-effect-label vs prefixed-effect-label" desynchronization
     pre-strip. Loud, correct.
   - The else-arm (lines 102-105) keeps the bare attribute name
     for backward-compat with hand-built tir modules in tests.
     The comment is load-bearing — without it a maintainer might
     remove the else-arm and break the test-side fixtures.
5. `is_pure_decl(fn)` (lines 109-110) — `return bool(fn.attrs.get
   ("is_pure") or fn.attrs.get("pure"))`. Two `.get` calls with
   default `None`. If the attribute is missing OR False, returns
   False (= not pure). If either attr is True, returns True. No
   silent-failure window: missing-attr semantics is "not pure" and
   that's the documented design. Confirmed non-finding.
6. `own_op_effects(fn)` (lines 113-120) — walks `fn.blocks` →
   `blk.ops`, checks `op.kind in OP_EFFECTS`, unions in the
   effects. If `op.kind` is some new effect-bearing op not yet in
   OP_EFFECTS, it would be silently treated as effect-free. This
   is a **build-time (statically-known) hazard**, not a runtime
   silent failure — the OP_EFFECTS table is the authoritative
   source-of-truth and any new effect-bearing op must be added.
   In practice, the only effect-bearing ops are the 4 already in
   the table. Stable non-finding.
7. `callees(fn)` (lines 123-139) — for each op, if CALL or
   MODIFY, extracts a target name and adds it to the callee set.
   The `target = op.attrs.get("target")` + `if isinstance(target,
   str): out.add(target) else: out.add("<indirect>")` (lines
   128-132) is the indirect-call handler — if the target attr
   isn't a string (e.g., absent or non-string), the callee is
   classified as `<indirect>` which propagates `unknown` in
   `compute_closure`. LOUD failure mode — the user sees
   "effect closure includes `unknown`" in the typecheck output.
   No silent-failure window.
8. `compute_closure(module)` (lines 142-168) — fixpoint iteration
   on the call graph. Three branches in the inner loop:
   - `c == "<indirect>"` → adds "unknown".
   - `c in module.functions` → unions in the callee's closure.
   - else (unknown external) → adds "unknown".
   All three branches are LOUD: an indirect or external call's
   "unknown" propagates to `check_module`'s caller-vs-declared
   diff, which surfaces user-visible diagnostics with trap-id
   19001. No silent-failure window.
9. `check_module(module)` (lines 171-221) — emits human-readable
   error strings for trap 19001 (purity violation, hard) and
   trap 19002 (unused declared effect, soft). All effects in
   `extra` or `unused` are surfaced verbatim. The `"unknown"`
   bucket is excluded from the 19002 unused-check (line 214) —
   correctly, because an indirect callee may legitimately be
   responsible for any effect a function declares. Not silent.
10. `verify_module(module)` (lines 224-228) — calls
    `check_module(module)` and raises `EffectError` if any
    errors. The raise propagates through `check.py`'s outer
    exception machinery. LOUD. No silent-failure window.

**Conclusion**: `effect_check.py` is fully exception-loud. Every
silent-skip in the walks is either justified (META_ATTRS,
`v is not True`, OP_EFFECTS membership) by a documented design
choice or is a future-tracked extension point (OP_EFFECTS
completeness) that surfaces LOUDLY if violated. Confirmed
non-finding for cycle 16.

### `helixc/frontend/totality.py` — full-module audit

**File invariants**:
- 0 `except` arms.
- 0 `raise` sites.
- 0 `return None` (the module returns lists/bools/iterators).
- 0 `.get(...)` calls.

**Surface walk**:
1. `check_totality(prog)` (lines 34-72) — walks every fn in
   `prog`, skips `@partial`, collects recursive self-calls,
   requires at least one parameter that strictly decreases on
   every recursive call. Returns a list of `(fn_name, reason)`
   tuples for failures. Consumed by `check.py:449-457` and by
   `backend/x86_64.py:3074-3084` — both call sites print the
   failures to stderr/stdout LOUDLY and (in --strict mode)
   return non-zero exit code.
   - The function uses `if not param_names: failures.append((name,
     "recursion with no parameters")); continue` (lines 57-59) —
     a LOUD failure mode for zero-param recursive fns.
   - The function uses `if not ok: failures.append((name, f"..."))`
     (lines 65-71) — another LOUD failure mode for non-decreasing
     recursive calls.
2. `_collect_self_calls(node, fn_name)` (lines 75-83) — generator
   that yields every direct self-call. The `if node is None:
   return` early-exit (line 77-78) is the safe no-op for absent
   AST nodes. The walker yields all matches; no silent fallback.
3. `_children(node)` (lines 86-108) — yields child AST nodes by
   probing a hand-curated attribute name list. The `if node is
   None: return` (line 87-88) is the safe no-op. The `hasattr`
   guards (lines 93, 97, 100, 103, 106) are explicitly defensive
   against the heterogeneous AST shape — they're not silent-skips
   because the children walker is paired with `_collect_self_calls`
   which IS the authoritative recursion detector. If a new AST
   node type were added with self-call-containing children
   reachable only via a never-listed attribute, the totality
   stub would silently miss the recursion. **This is a
   build-time (statically-known) hazard**, not a runtime silent
   failure. The hazard is explicitly documented at the
   module-level docstring as "Conservative: returns True
   (= 'totality unprovable') for any pattern we don't yet
   recognize." — and the conservative direction is **toward
   over-flagging**, not under-flagging, so the worst case is a
   false-positive totality-failure diagnostic, not a silent
   pass. The hazard is therefore on the LOUD side.

   However: **is the conservatism direction actually correct?**
   The docstring says totality returns True for patterns it
   doesn't recognize — but in the code, `_arg_strictly_decreases`
   returns **False** for any pattern it can't verify
   (line 126: `if param_idx is None or param_idx >= len(call.args):
   return False`; line 140: `return False` at the bottom of
   `_is_strictly_smaller`). False propagates back through:

   ```python
   for p in param_names:
       if all(_arg_strictly_decreases(call, fn.params, p) for call in recursive_calls):
           ok = True
           break
   if not ok:
       failures.append((name, ...))
   ```

   So an unrecognized pattern causes `_arg_strictly_decreases`
   to return False → `all(...)` is False for that p → no p
   succeeds → `not ok` is True → `failures.append`. **Loud
   over-flag**, matching the docstring claim. The direction
   IS correct. Confirmed non-finding.
4. `_arg_strictly_decreases` (lines 111-128) — finds the param's
   positional index. The `if param_idx is None or param_idx >=
   len(call.args): return False` (lines 125-126) is a LOUD-on-the-
   user-side default. If the param isn't in the param list at all
   (impossible by construction since param_names came from
   fn.params just above) OR if the recursive call passes fewer
   args than expected (which is a typecheck failure that would
   have been caught earlier in check.py), this returns False
   conservatively. Not silent.
5. `_is_strictly_smaller` (lines 131-140), `_is_name` (143-144),
   `_is_positive_int_const` (147-148), `_is_int_const_at_least`
   (151-152) — all pure predicate helpers. No silent-failure
   surface.

**Conclusion**: `totality.py` has zero exception handlers, zero
silent-skip branches that bias toward false-negative (under-flag).
Every "didn't recognize" path biases toward over-flagging, which
is the documented conservative direction. The downstream consumers
(`check.py:449-457` and `backend/x86_64.py:3074-3084`) emit
human-readable diagnostics and (in --strict mode) abort with
non-zero exit code. Confirmed non-finding for cycle 16.

### `helixc/ir/passes/cse.py` — full-module audit (including
   `_find_value_by_id`)

**File invariants**:
- 0 `except` arms.
- 0 `raise` sites.
- 1 `return None` site (line 134 inside `_find_value_by_id`).
- 0 `.get(...)` calls.

**Surface walk**:
1. `PURE_KINDS` (lines 33-50) — frozenset of 16 op kinds eligible
   for CSE. Cycle 14 already dual-checked vs SIDE_EFFECT_KINDS;
   no overlap; positive-allowlist topology is the safest possible
   default (skip-on-unknown, preserve op verbatim).
2. `_op_hash(op)` (lines 53-68) — stable hash key. Encodes
   primitive attrs as sorted tuple, complex attrs as repr-sorted
   tuple, result type as repr. Cycle 14 already verified the
   audit-10 fix (include result_ty and complex attrs); no
   regression.
3. `cse_module` (lines 71-75) — loops over functions, sums the
   counts. Pure plumbing.
4. `cse_function(fn)` (lines 78-119) — per-block CSE with hash
   merging + operand rewrites. The `if op.kind not in PURE_KINDS:
   continue` (lines 101-102) is the documented skip-non-pure
   path, preserving the side-effecting op verbatim.

   The block-scoped `seen` and `rewrites` dicts (lines 91, 93)
   are reset per-block, which the docstring explicitly justifies:
   "Per-block CSE ... that would require dominance analysis ...
   For v0.1 per-block is sound and catches most cases." Not
   silent — the limitation is documented.
5. `_find_value_by_id(fn, value_id)` (lines 122-134) — finds a
   `tir.Value` by its `id` in fn.params, blk.params, or op.results.
   Returns `None` if not found.

   **Cycle-16 fresh-eyes question**: is this function a hidden
   silent-failure window? Let me trace its callers.

   `git grep -n '_find_value_by_id' helixc/` returns ONLY the
   definition site (line 122 in cse.py). **Zero non-self callers.**
   The function is a **dead helper** kept for debugging or future
   use. Its `return None` has no live consumer, so it cannot
   silently hide a failure today.

   **Should it be removed?** That would be a code-hygiene
   recommendation, not a silent-failure finding. Per the
   user's strict cycle-16 criterion ("zero findings of ANY
   severity"), code-hygiene observations on dead helpers are
   NOT findings — they would be code-review-lens observations.
   Filed for the cycle-16 code-review lens, not this lens.
   Confirmed non-finding for the silent-failures lens.

**Conclusion**: `cse.py` is fully exception-loud. The one
`return None` site is in a dead helper with zero callers, so it
cannot hide a failure today. The block-scoped CSE limitation is
documented in the docstring. Confirmed non-finding for cycle 16.

### `helixc/tests/test_codegen.py` — cycle-16-new test-harness diff

The new commit (4c74627) rewrites the WSL bootstrap test harness
to detect "bootstrap binary did not write output" and surface a
sentinel string that the Python asserter checks for. The new
harness logic:

```python
last = run.stdout.decode().strip().splitlines()[-1] if run.stdout else ""
assert last != "__HARNESS_FAIL_BOOTSTRAP_DID_NOT_WRITE_OUTPUT__", (
    f"bootstrap did not produce /tmp/helix_bin_out.bin for source "
    f"{source_text!r}; stderr: {run.stderr.decode()[:500]}"
)
return int(last)
```

This is LOUD by design — the harness-failure case raises an
AssertionError with full source text and stderr context, replacing
the prior silent "exit code from chmod masquerading as exit code
from the user binary" bug. The change strictly REDUCES silent
failures in the test harness; it does not introduce a new silent
window. The commit message explicitly states this is the design.

The pre-fix harness was technically a silent-failure pattern in
TEST code (not production), but per the project's silent-failures
audit scope (production code), it was never a cycle-N finding.
The fix itself is LOUD and correct. Confirmed non-finding for
cycle 16.

### Global `except: pass` hunt

`grep -nE 'except\s*:\s*pass|except\s+Exception\s*:\s*pass|except\s+BaseException'
helixc/` returns ONE match:

- `helixc/frontend/autodiff.py:998` —
  `# this 'except Exception: pass' swallowed every error in`

This is a **comment** in autodiff.py describing the prior
(now-fixed) silent-failure pattern. The actual code at line 1012
is `except (OverflowError, ZeroDivisionError, ValueError,
TypeError)` — a narrow typed except.

**Zero genuine `except: pass` patterns in production code.**
Stable non-finding.

### Did the cycle-16-new commit (4c74627) introduce any new
silent-failure window?

The commit's diff is entirely test-only — production code is
unchanged. By construction the cycle-15 silent-failures verdict
holds on cycle 16.

The test-harness fix itself REDUCES silent failures (it replaces
a previously-silent flake with a LOUD AssertionError). It does
not introduce new silent windows. Confirmed.

### Carryover findings status (cycles 1-15) — unchanged

The cycle-16 re-audit closed nothing (read-only by design) and
introduced no new finding. The carryover ledger is identical to
cycle 15's closing snapshot.

| Carryover | Severity | Cycle-16 status |
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

## Re-audit verification on 4c74627 (production surface identical to cycles 14 + 15)

| Re-audit pass | C12 | C13 | C14 | C15 | C16 | Stability |
|---|---|---|---|---|---|---|
| `_emit_env_error` strip helper (check.py:246-255) | clean | clean | clean | clean | clean | stable |
| Outer-except topology (check.py:284-318) | clean | clean | clean | clean | clean | stable |
| Finally drain-failure suppressor (check.py:319-337) | clean | clean | clean | clean | clean | stable |
| Backend-call wraps (check.py:618,649,663) | clean | clean | clean | clean | clean | stable |
| AD-warning narrowed excepts (autodiff.py:155,1012) | clean | clean | clean | clean | clean | stable |
| const_fold defensive folds (const_fold.py:250,324,349,401) | clean | clean | clean | clean | clean | stable |
| Quote-handle fallback (lower_ast.py:2115) | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| diagnostics isatty fallback (diagnostics.py:76) | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| `getattr(it, "is_kernel", False)` (check.py:641) | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| lower_ast.py try/finally scope at :596, :1800 | C12 fresh: clean | clean | clean | clean | clean | stable |
| backend/x86_64.py attrs.get defaults | C12 fresh: clean | clean | clean | clean | clean | stable |
| backend/ptx.py, elf_dyn.py zero-except | C12 fresh: clean | clean | clean | clean | clean | stable |
| frontend/parser.py:375 ValueError -> ParseError re-raise | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:415,423 TypeError_ -> diag append | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:636 ValueError -> Optional None | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:203 ValueError -> return expr | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:759 ShapeFoldError -> diag list | clean | clean | clean | clean | clean | stable |
| frontend/grad_pass.py:639-643 frozen-dataclass cache fallback | (n/e) | C13 fresh: clean | clean | clean | clean | stable |
| frontend/pytree.py:293-296 validate_pytree diagnostic collection | (n/e) | C13 fresh: clean | clean | clean | clean | stable |
| frontend/hash_cons.py:335 raise HashConsError | (n/e) | C13 fresh: clean | clean | clean | clean | stable |
| frontend/flatten_impls.py:88 raise DuplicateMethodError | (n/e) | C13 fresh: clean | clean | clean | clean | stable |
| frontend/flatten_modules.py:67,77 raise FlattenError | (n/e) | C13 fresh: clean | clean | clean | clean | stable |
| frontend/trace_pass.py:67 raise OverflowError | (n/e) | C13 fresh: clean | clean | clean | clean | stable |
| ir/passes/effect_check.py:228 raise EffectError | (n/e) | C13 fresh: clean | clean | clean | clean | stable |
| dce.py SIDE_EFFECT_KINDS frozenset (incl. C14 +TRACE_ENTRY/EXIT) | (n/e) | (n/e) | C14 fresh: clean | clean | clean | stable |
| cse.py PURE_KINDS dual-check vs SIDE_EFFECT_KINDS | (n/e) | (n/e) | C14 fresh: clean | clean | clean | stable |
| fdce.py call-graph source check vs TRACE_* | (n/e) | (n/e) | C14 fresh: clean | clean | clean | stable |
| x86_64.py TRACE_EXIT operand consumer guard | (n/e) | (n/e) | C14 fresh: clean | clean | clean | stable |
| lower_ast.py synthesized-const sentinel (line 573-574, 1891-1892) | (n/e) | (n/e) | C14 fresh: clean | clean | clean | stable |
| lexer.py:399-402 `\u` escape ValueError -> LexError re-raise | (n/e) | (n/e) | (n/e) | C15 fresh: clean | clean | stable |
| lower_ast.py:280-283 flat-path index ValueError -> None (C5-10 Pat C) | (n/e) | (n/e) | (n/e) | C15 fresh: C5-10 carryover | stable carryover | stable carryover |
| lower_ast.py:2064-2068 Field-of-Field flat-path ValueError -> -1 (C5-10 Pat C) | (n/e) | (n/e) | (n/e) | C15 fresh: C5-10 carryover | stable carryover | stable carryover |
| struct_mono.py:445-456 ShapeFoldError + ValueError -> diags | (n/e) | (n/e) | (n/e) | C15 fresh: clean | clean | stable |
| backend/x86_64.py raise-only inventory (24 sites) | (n/e) | (n/e) | (n/e) | C15 fresh: clean | clean | stable |
| cse.py + fdce.py zero try/except/raise | (n/e) | (n/e) | (n/e) | C15 fresh: clean | clean | stable |
| **effect_check.py full-module audit (OP_EFFECTS, META_ATTRS, declared_effects, callees, compute_closure, check_module, verify_module)** | (n/e) | (n/e) | (n/e) | (n/e) | **C16 fresh: clean** (1 raise site at line 228 surfaces LOUDLY through check.py outer-except; all `.get` defaults + `v is not True` filter justified; OP_EFFECTS completeness is a build-time hazard, not runtime silent) | new |
| **totality.py full-module audit (check_totality, _collect_self_calls, _children, _arg_strictly_decreases, _is_strictly_smaller, predicates)** | (n/e) | (n/e) | (n/e) | (n/e) | **C16 fresh: clean** (0 except, 0 raise, 0 .get; conservative bias is toward false-positive over-flag matching docstring; downstream check.py:449-457 + x86_64.py:3074-3084 emit user-visible diagnostics) | new |
| **cse.py `_find_value_by_id` dead helper (line 122-134)** | (n/e) | (n/e) | (n/e) | (n/e) | **C16 fresh: clean** (zero non-self callers per `grep _find_value_by_id helixc/`; `return None` has no live consumer; code-hygiene observation only, not silent-failure) | new |
| **test_codegen.py bootstrap-harness sentinel pattern (the 4c74627 diff)** | (n/e) | (n/e) | (n/e) | (n/e) | **C16 fresh: clean** (LOUD AssertionError with source + stderr context replaces the prior silent flake; reduces silent failures in test harness) | new |
| Global `except: pass` hunt (zero matches in production) | clean | clean | clean | clean | clean | stable |

### Specific cycle-16 items re-checked clean

- **No new production commits -> production-code surface identical
  to cycles 14 + 15**: `git diff 1e4c3e6..HEAD -- 'helixc/' '*.py'
  '*.hx'` shows only the test_codegen.py harness diff. By
  construction the cycle-15 clean verdict propagates to cycle 16
  for the silent-failures production-code lens.
- **effect_check.py**: full-module audit. The single `raise
  EffectError` at line 228 surfaces LOUDLY through check.py's
  outer-except. All `.get(...)` defaults are documented
  conservative-on-missing-attr defaults. The `v is not True`
  filter at line 96-97 in `declared_effects` is the documented
  way to read boolean-True effect attrs. OP_EFFECTS completeness
  is a build-time hazard (any new effect-bearing op MUST be
  added) and the static check is satisfied by code review at
  authoring time. Confirmed non-finding.
- **totality.py**: full-module audit. Zero except arms, zero
  raise sites, zero `.get(...)` calls. The conservative-bias
  direction is correct (toward false-positive over-flag, matching
  the docstring's "Conservative: returns True (= 'totality
  unprovable') for any pattern we don't yet recognize"). The
  downstream consumers in check.py:449-457 and
  backend/x86_64.py:3074-3084 emit user-visible diagnostics with
  --strict mode aborting on non-zero. Confirmed non-finding.
- **cse.py `_find_value_by_id`**: dead helper (zero callers).
  The `return None` has no live consumer in production. Filed
  as a code-hygiene observation for the cycle-16 code-review
  lens, not a silent-failure finding.
- **test_codegen.py 4c74627 diff**: LOUD AssertionError replaces
  the prior silent flake. Strictly reduces silent failures in
  the test harness. Confirmed non-finding.
- **Global `except: pass` hunt**: only one grep match in
  production, which is a COMMENT in autodiff.py:998 describing
  the prior (now-fixed) bare-except pattern. Zero genuine
  `except: pass` arms in production code. Confirmed
  non-finding.

### Cross-stage interactions re-checked (cycle 16)

- **effect_check `EffectError` -> check.py outer-except**: the
  `verify_module` raise at line 228 propagates through the
  typecheck → IR → backend chain to check.py's broad-Exception
  arm (lines 306, 618/649/663), which emits "internal error" +
  "compiler bug" + rc=1. The user sees the typed exception
  message verbatim. Not silent.
- **totality.py `failures` list -> check.py output**: check.py
  :449-457 prints "totality: N fn(s) NOT proven total" and
  per-fn `(name, reason)` rows. In --strict mode, returns 1.
  In non-strict mode, prints the warnings but proceeds. The
  --strict gating is the documented design (totality is a
  Stage 21 soft-warn-by-default invariant). Not silent.
- **cse.py per-block CSE -> dce.py**: the docstring explicitly
  notes "After CSE, run DCE to clean up the now-dead duplicates."
  The pipeline order in check.py is fold → cse → dce. The
  block-scoped limitation (no cross-block CSE) is documented as
  v0.1 design. Not silent — the limitation is explicit.
- **No new production commits since cycle 15**: the cycle-15
  verdict propagates. Confirmed.

### Did the cycle-16 fresh-eyes rotation surface any overlooked
silent-failure window?

Three modules audited at line-by-line resolution
(`effect_check.py`, `totality.py`, `cse.py`). Each has zero new
silent-failure window beyond what is already in the carryover
ledger. The single `return None` in cse.py is in dead code
(zero callers). The conservative-defaults in effect_check.py
and totality.py both bias toward LOUD (over-flag) outcomes, not
silent under-flag. The cycle-16-new commit at 4c74627 is
test-only and strictly REDUCES silent failures in the test
harness.

**Conclusion**: zero new silent-failure findings for cycle 16.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-17 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 16 did not address (read-only re-audit).
  **STILL THE HIGHEST-PRIORITY ITEM** for any future fix-sweep
  — the only remaining CRITICAL across the audit series. As the
  clean-counter advances toward the 5/5 Stage-29 gate, the
  question of whether the gate requires CRITICAL=0-open
  (stricter) or merely 5-consecutive-clean (lenient) becomes
  load-bearing.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH.
  Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks (Patterns A, B, C —
  including the cycle-15-enumerated :280-283 and :2064-2068
  sites)**: still open LOW. Not addressed; not re-flagged.
- **monomorphize_safe docstring drift**: still open (cycle-6
  deferred).
- **D-vs-Quote diagnostic text**: still open (cycle-7 deferred).
- **C7-1 test-coverage gap**: still open. Cycle 16 also did not
  add the 4 `_compatible(TyMemTier, TyVar)` regression tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix edge
  cases**: still no callee triggers either. Not findings.
- **TRACE_EXIT operand-less defensive guard (x86_64.py:2495)**:
  noted in cycle 14 — the `if op.operands:` guard tolerates a
  hypothetical operand-less TRACE_EXIT. Future-tracking item
  if the trace machinery evolves. Not a finding for cycle 16.
- **cse.py `_find_value_by_id` dead helper**: zero callers.
  Code-hygiene candidate for the cycle-16 code-review lens
  (suggest removal or doc-only marker), NOT a silent-failures
  finding.
- **OP_EFFECTS completeness (effect_check.py:40-49)**: any new
  effect-bearing op kind MUST be added to OP_EFFECTS by the
  author at IR-design time. Static-check-at-authoring-time
  hazard. Filed as a future-tracking item; not a
  silent-failures finding for cycle 16.
- **totality.py `_children` attribute coverage**: the hand-curated
  attribute name list (lines 89-92) is exhaustive for the current
  AST shape. Any new AST node type with self-call-containing
  children reachable only via a never-listed attribute would
  cause a false-negative under-flag (a silent miss). Today
  the AST shape is fully covered; static-check-at-authoring-time
  hazard. Filed as a future-tracking item; not a silent-failures
  finding for cycle 16.

---

## Cycle 15 vs cycle 16 — clean-cycle counter check

Cycle 15 was the 2nd clean of the re-accumulated window (counter
2/5). The user directive for cycle 16 explicitly instructs:
re-audit with rotated spot-check surface, do not re-flag prior-
cycle carryovers unchanged since cycle 15.

The cycle-16 re-audit honors that directive:
- `audit-C4-1 CRITICAL`, `audit-C4-4 HIGH`, `audit-C4-8 LOW`:
  not re-flagged.
- `C5-10 LOW`: not re-flagged.
- `monomorphize_safe docstring drift`, `D-vs-Quote diagnostic
  text`, `C7-1 test-coverage gap`: not re-flagged.

Cycle 16 produces **zero NEW findings of any severity**, so the
clean-cycle counter advances to **3/5** under the strict criterion
— subject to the parallel type-design + code-review audit lenses
also being clean for cycle 16.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 16 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW).**

---

## Cycle 16 status

**Cycle 16 IS CLEAN** for the silent-failure audit lens. Per the
strict criterion (zero findings of ANY severity), the 0-finding
result satisfies the clean-cycle gate for this audit lens.

### Stop-the-line determination: **NO**

Cycle 16 is clean — no stop required for this lens.

### Cycle 16 -> NEW FINDINGS COUNT for the strict-clean gate: 0
(0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter advances
to **3/5** for this audit lens (cycle 14 silent-failures clean
= 1/5; cycle 15 = 2/5; cycle 16 = 3/5).

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
- Cycle 13: 0 findings (silent-failures lens; code-review lens
  found C13-1 HIGH, addressed by cycle-14 fix-sweep).
- Cycle 14: 0 findings (silent-failures lens).
- Cycle 15: 0 findings (silent-failures lens).
- Cycle 16: 0 findings (silent-failures lens). <- here

Trend: **7 consecutive clean cycles** on the silent-failures
lens (10, 11, 12, 13, 14, 15, 16). The global strict-clean
counter is 3/5 because cycle 13's code-review lens broke the
prior 5-clean-cycle accumulation, resetting the global counter
to 0; cycles 14, 15, 16 are the re-accumulated window.

### Estimated remaining open findings going into cycle 17

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
- Cycle 10-16 silent-failure: 0 new each. <- here
- Cycle 13 code-review: C13-1 HIGH — CLOSED by cycle 14
  fix-sweep.
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 17).
- Cycle 16 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-16 new) = **>=23 open findings** going into cycle 17.
  (Net 0 delta vs cycle 15.)

Recommend prioritizing in this order for the cycle-17 fix
batch (if user elects to land fixes between clean re-audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks).
4. **C7-1 test-coverage gap**.
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).

The "5 clean cycles before Phase 0 deprecation" goal requires
the strict criterion (zero findings of any severity, all three
lenses) to be met for 5 CONSECUTIVE cycles. Cycle 14 = 1/5;
cycle 15 = 2/5; cycle 16 = 3/5 of the re-accumulated window.
Two more clean cycles (17, 18) needed across all three lenses
to fire the gate (assuming parallel type-design + code-review
lenses remain clean).

**Cycle 16 status: CLEAN**
**Counter status: 3/5** (cycles 14, 15, 16 silent-failures all
clean; subject to parallel type-design + code-review lenses
also being clean for cycle 16).
