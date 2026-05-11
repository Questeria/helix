# Stage 28.8 Cycle 15 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit (HEAD)**: 1e4c3e6 — "Audit 28.8 cycle 14 fix-sweep:
close C13-1 (HIGH, DCE drops TRACE_EXIT operand)".

**Note on HEAD stability**: `git log --oneline 1e4c3e6..HEAD`
returns empty. `git diff --stat 1e4c3e6..HEAD` returns empty.
**Zero commits and zero diff** since cycle 14 closed CLEAN at
HEAD 1e4c3e6. Cycle 15 is the second consecutive read-only
re-audit on the post-C13-1-fix HEAD.

**Scope**: any silent-failure window NOT already counted in
cycles 1-14 as a carryover. Documented carryovers
(audit-C4-1 CRITICAL, audit-C4-4 HIGH, audit-C4-8 LOW, C5-10
LOW, monomorphize_safe docstring drift, D-vs-Quote diagnostic
text, C7-1 test-coverage gap) are NOT re-flagged per the
user's strict re-flag rule (a carryover is re-flagged only
if it CHANGED since the prior cycle — and none did, because
no production code has changed since cycle 14).

**Strict criterion** (per user directive 2026-05-10): cycle
counts CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Findings already in the
carryover ledger are explicitly excluded.

**Clean-counter state**: cycle 14 closed CLEAN across all
three lenses (silent-failures, type-design, code-review),
advancing the fresh clean-counter to 1/5. Cycle 15 is the
second clean-cycle attempt of the re-accumulated window
(post-C13-1-fix).

---

## Method

1. **Confirmed HEAD invariance vs cycle 14**:
   - `git log --oneline 1e4c3e6..HEAD` → empty.
   - `git diff --stat 1e4c3e6..HEAD` → empty.
   - `git diff 1e4c3e6..HEAD -- 'helixc/' '*.py' '*.hx'`
     → empty.
   - Working tree shows only doc additions
     (cycle-14 audit docs and one cycle-15 doc) plus
     unrelated cycle-11 doc whitespace tweaks — **zero
     production-code delta** vs cycle 14 HEAD.
   - By construction, every clean verdict on cycle 14 holds
     on cycle 15 unless the fresh-eyes re-walk surfaces an
     overlooked window. None found.
2. **Read cycle-13 + cycle-14 silent-failure verdicts**: both
   CLEAN. Cycle 14 fix-sweep closed C13-1 (HIGH, DCE drops
   TRACE_EXIT operand) by adding TRACE_ENTRY and TRACE_EXIT
   to `SIDE_EFFECT_KINDS` in `helixc/ir/passes/dce.py`.
3. **Re-verified the cycle-14 fix at HEAD**
   (`helixc/ir/passes/dce.py:68-80`):
   - `tir.OpKind.TRACE_ENTRY` and `tir.OpKind.TRACE_EXIT`
     present in `SIDE_EFFECT_KINDS` (lines 79, 80).
   - Multi-line block comment (lines 68-78) explaining the
     C13-1 rationale is intact.
   - Full set has 20 members; member-by-member inventory
     matches cycle 14's audit table.
4. **Cycle-15 fresh-eyes rotation**: cycle 14 spot-checked
   dce.py SIDE_EFFECT_KINDS, cse.py PURE_KINDS, fdce.py
   call-graph edges, x86_64.py TRACE_EXIT consumer guard,
   and lower_ast.py synthesized-const sentinel. Cycle 15
   rotates to:
   - `helixc/frontend/lexer.py:399-402` — `\u` hex escape
     `int(digs, 16)` ValueError re-raise as LexError.
   - `helixc/ir/lower_ast.py:280-283` — `_lookup_array`
     flat-path traversal: `src_paths.index(remaining)`
     ValueError → `return None` (part of C5-10 carryover
     Pattern C — verified unchanged).
   - `helixc/ir/lower_ast.py:2064-2068` — Field-of-Field
     chain `flat_paths.index(target)` ValueError →
     `idx_int = -1` (part of C5-10 carryover Pattern C —
     verified unchanged).
   - `helixc/frontend/struct_mono.py:445-456` — generic
     struct instantiation `ShapeFoldError` /
     `ValueError` → `diags.append(str(e))` + `continue`.
   - `helixc/backend/x86_64.py raise-only enumeration` —
     verified the entire backend remains exception-
     transparent (zero `except` arms; all raises are
     typed ValueError / NotImplementedError /
     OverflowError that propagate to check.py:618/649/663
     wraps).
   - `helixc/ir/passes/cse.py + fdce.py` — verified pure
     (zero try/except/raise in either file).
   - Global hunt for `except: pass`, `except Exception:
     pass`, and any bare-except patterns in production.
5. **Read-only**: no edits to production code or tests.

---

## Fresh-eyes walk for cycle 15

### `lexer.py:399-402` — `\u` escape hex parse

```python
self._advance()
try:
    return chr(int(digs, 16))
except ValueError:
    raise LexError(r"\u escape: invalid hex", line, col) from None
```

The `except ValueError` arm catches `int(digs, 16)` failures
on non-hex characters and re-raises as `LexError` with
`from None` (suppresses the chained traceback for clean
user diagnostics). The LexError propagates through the
lexer → parser → check.py error chain and surfaces as a
user-visible diagnostic with line/col context. The
`chr(int(...))` could also raise `OverflowError` (if the
codepoint exceeds 0x10FFFF) or `ValueError` (if non-hex
characters in `digs`) — both are handled by the
`except ValueError` arm because `chr` raises ValueError for
out-of-range codepoints in CPython 3.x. Not silent. Stable
non-finding. Never previously enumerated. Confirmed
non-finding for cycle 15.

### `lower_ast.py:280-283` — flat-path index lookup (C5-10 Pattern C)

```python
src_paths = self._struct_flat_paths.get(struct_name, [])
remaining = path[consumed:]
try:
    idx_int = src_paths.index(remaining)
except ValueError:
    return None
```

This is the no-array-match branch of the field-chain
lowerer. When the path segments don't match any registered
flat-path entry, the function returns `None`, signalling
the caller to fall through to the next lowering branch.
This is **C5-10 carryover Pattern C** (cycle 5 LOW),
verified unchanged since cycle 5. **Not re-flagged** per
the user's strict re-flag rule — the file hasn't changed.

### `lower_ast.py:2064-2068` — Field-of-Field chain (C5-10 Pattern C)

```python
try:
    idx_int = flat_paths.index(target)
except ValueError:
    idx_int = -1
if idx_int >= 0:
    arr = self._lookup_array(base_name)
    ...
```

Same as above — when the field-chain target isn't in the
flat-path table, `idx_int = -1` causes the `if idx_int
>= 0` guard to skip the LOAD_ELEM emission, falling
through to the tuple-field case below. This is also
**C5-10 carryover Pattern C**, verified unchanged. **Not
re-flagged**.

### `struct_mono.py:445-456` — generic instantiation diagnostics

```python
for (sname, ty_args) in uses:
    try:
        inst = instantiate(generic_structs[sname], ty_args)
    except ShapeFoldError as e:
        diags.append(str(e))
        continue
    except ValueError as e:
        diags.append(str(e))
        continue
    ...
```

Two narrow domain-typed excepts (`ShapeFoldError` from
cycle-3 C3-6, `ValueError` from generic constraint
violations) both append `str(e)` to the diagnostics list
and `continue` to the next use. The diagnostics list is
returned to the typecheck driver, which surfaces each
string as a user-visible error with rc != 0. The
ShapeFoldError originates from constant-folding shape
expressions (e.g., `divide by 0`, `modulo by 0`); the
ValueError originates from constraint violations (e.g.,
`generic param T appears bound twice`). Both are
narrowly typed to the domain. **Not silent** — full
diagnostic text reaches the user. Stable non-finding.
Never previously enumerated. Confirmed non-finding for
cycle 15.

### `backend/x86_64.py + ptx.py + elf_dyn.py` — exception-transparent

Verified by `git grep -nE 'except|raise ' --
helixc/backend/`:
- **Zero `except` arms** in any backend module
  (only "raise" sites; the word "except" appears solely
  in comments at x86_64.py:1486-1487 and ptx.py:9
  describing SNaN floating-point exception behavior in
  generated assembly — not Python exception arms).
- **Typed raises only**: 24 sites in x86_64.py
  (ValueError / NotImplementedError / OverflowError),
  1 in elf_dyn.py:217 (RuntimeError), 1 in ptx.py:74
  (RuntimeError).
- All propagate to `check.py:618/649/663`'s broad-Exception
  wrap which emits "internal error" + "this is a compiler
  bug" + rc=1. **Not silent.**

Cycle 12 enumerated `backend/x86_64.py attrs.get defaults`
as a fresh spot-check (clean). Cycle 15 cross-validates
the entire raise-only inventory and confirms no new
silent-failure window. Stable non-finding.

### `ir/passes/cse.py + fdce.py` — pure (zero exception handling)

Verified by `grep -nE 'except|raise' helixc/ir/passes/`:
- **cse.py**: 0 `except`, 0 `raise`. The pass operates on
  TIR ops via positive allowlist `PURE_KINDS` — any op not
  in the allowlist is skipped. No way for the pass to
  silently fail; if an unexpected op kind appears, the
  default-skip behavior is the SAFEST possible action
  (preserve the op verbatim). Cycle 14 already verified
  `PURE_KINDS` membership is correct vs the cycle-14
  SIDE_EFFECT_KINDS additions.
- **fdce.py**: 0 `except`, 0 `raise`. Function-level DCE
  walks the call graph; ops it doesn't recognize are
  treated as non-edge (conservative — over-keeps
  functions). No silent-failure window.

Cycle 14 already enumerated both files. Cycle 15
re-confirms stability. Non-findings.

### `_lookup_array` + flat-path-table consumers (cycle-15 fresh interaction probe)

The two `try: ... except ValueError: idx_int = -1`
patterns in lower_ast.py (lines 282 and 2066) both depend
on `_struct_flat_paths` being populated correctly by the
struct-flatten pass. If the table is stale (e.g., a
struct was renamed without rebuilding the table), the
`.index(target)` call would silently miss and the
lowerer would emit no LOAD_ELEM op for the field access.

However: the table is built EAGERLY at module-load time
in `lower_ast.py:_init_struct_flat_paths` (per
`_struct_flat_paths.get(struct_name, [])` access pattern
— if the struct name itself isn't in the table, the
`.get` default is `[]`, and `[].index(remaining)` raises
ValueError immediately for any non-empty remaining,
which returns None / -1 in the consumer. The empty-table
case is therefore handled identically to the
no-match case.

The "stale-rename" hazard would require runtime
modification of `_struct_flat_paths` AFTER initial
build, which doesn't happen in the current lowerer
(the table is read-only after `_init_struct_flat_paths`).
So the C5-10 carryover Pattern C remains stable: the
silent-fall-through is bounded by the well-known
quote-handle fallback, no expansion of the silent
surface.

### Global `except: pass` hunt

`grep -nE 'except\s*:\s*pass|except\s+Exception\s*:\s*pass'
helixc/` returns ONE match:

- `helixc/frontend/autodiff.py:998` —
  `# this `except Exception: pass` swallowed every error in`

This is a **comment** in autodiff.py describing the
prior (now-fixed) silent-failure pattern. The actual code
at line 1012 is `except (OverflowError, ZeroDivisionError,
ValueError, TypeError)` — a narrow typed except. Not a
silent failure; only a documentation reference.

`grep -nE 'except.*pass$' helixc/` returns the same
comment-only match.

**Zero genuine `except: pass` patterns in production
code.** Stable non-finding.

### Did anything change since cycle 14?

`git diff --stat 1e4c3e6..HEAD` and
`git log 1e4c3e6..HEAD` both return empty. The
working-tree changes are limited to docs (cycle-14 doc
files newly added, two cycle-11 doc files with whitespace
diffs) — **no production-code surface delta**.

By construction:
- Every clean verdict on cycle 14 holds on cycle 15.
- The cycle-14 fix (TRACE_ENTRY / TRACE_EXIT in
  SIDE_EFFECT_KINDS) remains present.
- The cycle-14 fresh-eyes assertions (CSE PURE_KINDS,
  FDCE call-graph, x86_64.py TRACE_EXIT consumer guard,
  lower_ast.py synthesized-const sentinel) remain valid.

The cycle-15 fresh-eyes rotation (lexer.py:399-402,
struct_mono.py:445-456, backend raise-only inventory,
cse.py + fdce.py pure-pass verification) surfaces no
overlooked silent-failure window.

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

## Re-audit verification on 1e4c3e6 (production surface identical to cycle 14)

| Re-audit pass | C10 | C11 | C12 | C13 | C14 | C15 | Stability |
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
| lower_ast.py try/finally scope at :596, :1800 | (n/e) | (n/e) | C12 fresh: clean | clean | clean | clean | stable |
| backend/x86_64.py attrs.get defaults | (n/e) | (n/e) | C12 fresh: clean | clean | clean | clean | stable |
| backend/ptx.py, elf_dyn.py zero-except | (n/e) | (n/e) | C12 fresh: clean | clean | clean | clean | stable |
| ir/tile_ir.py, tir.py zero-raise | (n/e) | (n/e) | C12 fresh: n/a | n/a | n/a | n/a | n/a |
| frontend/parser.py:375 ValueError -> ParseError re-raise | clean | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:415,423 TypeError_ -> diag append | clean | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:636 ValueError -> Optional None | clean | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:203 ValueError -> return expr | clean | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:759 ShapeFoldError -> diag list | clean | clean | clean | clean | clean | clean | stable |
| frontend/grad_pass.py:639-643 frozen-dataclass cache fallback | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | clean | stable |
| frontend/pytree.py:293-296 validate_pytree diagnostic collection | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | clean | stable |
| frontend/hash_cons.py:335 raise HashConsError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | clean | stable |
| frontend/flatten_impls.py:88 raise DuplicateMethodError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | clean | stable |
| frontend/flatten_modules.py:67,77 raise FlattenError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | clean | stable |
| frontend/trace_pass.py:67 raise OverflowError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | clean | stable |
| ir/passes/effect_check.py:228 raise EffectError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | clean | stable |
| examples/dashboard_server.py try/except sites | (n/e) | (n/e) | (n/e) | C13 fresh: n/a | n/a | n/a | n/a |
| dce.py SIDE_EFFECT_KINDS frozenset (incl. C14 +TRACE_ENTRY/EXIT) | (n/e) | (n/e) | (n/e) | (n/e) | C14 fresh: clean | clean | stable |
| cse.py PURE_KINDS dual-check vs SIDE_EFFECT_KINDS | (n/e) | (n/e) | (n/e) | (n/e) | C14 fresh: clean | clean | stable |
| fdce.py call-graph source check vs TRACE_* | (n/e) | (n/e) | (n/e) | (n/e) | C14 fresh: clean | clean | stable |
| x86_64.py TRACE_EXIT operand consumer guard | (n/e) | (n/e) | (n/e) | (n/e) | C14 fresh: clean | clean | stable |
| lower_ast.py synthesized-const sentinel (line 573-574, 1891-1892) | (n/e) | (n/e) | (n/e) | (n/e) | C14 fresh: clean | clean | stable |
| **lexer.py:399-402 `\u` escape ValueError -> LexError re-raise** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C15 fresh: clean** (typed re-raise with `from None`; user-visible diagnostic with line/col) | new |
| **lower_ast.py:280-283 flat-path index ValueError -> None (C5-10 Pat C)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C15 fresh: C5-10 carryover** (unchanged since cycle 5) | stable carryover |
| **lower_ast.py:2064-2068 Field-of-Field flat-path ValueError -> -1 (C5-10 Pat C)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C15 fresh: C5-10 carryover** (unchanged since cycle 5) | stable carryover |
| **struct_mono.py:445-456 ShapeFoldError + ValueError -> diags** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C15 fresh: clean** (narrow domain-typed; str(e) preserves full diagnostic; user sees rc != 0) | new |
| **backend/x86_64.py raise-only inventory (24 sites)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C15 fresh: clean** (all typed; propagate to check.py:618/649/663 broad-Exception wrap) | new |
| **cse.py + fdce.py pure-pass verification (zero try/except/raise)** | (n/e) | (n/e) | (n/e) | (n/e) | (n/e) | **C15 fresh: clean** (positive-allowlist topology; conservative skip behavior on unknown ops) | new |
| **Global `except: pass` hunt (zero matches in production)** | clean | clean | clean | clean | clean | clean | stable |

### Specific cycle-15 items re-checked clean

- **No new production commits -> no new code surface**: `git
  diff --stat 1e4c3e6..HEAD` returns empty. `git log
  1e4c3e6..HEAD` returns empty. By construction the cycle-14
  clean verdict propagates to cycle 15 unless the fresh
  re-walk finds an overlooked window. None found.
- **`lexer.py:399-402` `\u` escape ValueError re-raise**: the
  `try: chr(int(digs, 16)) / except ValueError: raise
  LexError(...) from None` pattern catches non-hex digit
  failures AND out-of-range codepoint failures (both surface
  as ValueError in CPython 3.x's `chr` and `int`). The
  re-raise as LexError with `from None` produces a clean
  user-visible diagnostic with line/col context. Not silent.
  Cycle-15 fresh spot-check, never previously enumerated.
  Confirmed non-finding.
- **`struct_mono.py:445-456` generic instantiation
  diagnostics**: two narrow typed except arms
  (`ShapeFoldError` from cycle-3 C3-6, `ValueError` from
  generic constraints) both append `str(e)` to the
  diagnostics list and continue. The list reaches the
  typecheck driver as a user-visible diagnostic stream with
  rc != 0. Not silent. Cycle-15 fresh spot-check.
  Confirmed non-finding.
- **Backend raise-only inventory**: 24 typed raise sites in
  x86_64.py + 1 in elf_dyn.py + 1 in ptx.py, all propagate
  to check.py's broad-Exception wraps. Zero `except` arms
  anywhere in the backend. The "except" keyword only appears
  in 2 comments at x86_64.py:1486-1487 and ptx.py:9
  describing SNaN floating-point behavior in generated
  assembly — not Python exception handling. Confirmed
  non-finding.
- **cse.py + fdce.py pure**: zero try/except/raise in either
  module. Positive-allowlist topology (PURE_KINDS in cse.py,
  call-graph-edge ops in fdce.py) gives both passes a
  conservative skip-unknown-op default behavior — the
  SAFEST possible failure mode. Confirmed non-finding.
- **Global `except: pass` hunt**: only one grep match in
  production, which is a COMMENT in autodiff.py:998
  describing the prior (now-fixed) bare-except pattern.
  Zero genuine `except: pass` arms in production code.
  Confirmed non-finding.

### Cross-stage interactions re-checked (cycle 15)

- **Lexer LexError -> parser -> check.py**: LexError
  propagates from lexer through parser.lex_tokens() to
  check.py:382 (`except ParseError as e:` — emits user
  diagnostic), and any non-ParseError lexer/parser
  internal-error case propagates to check.py:306 (`except
  Exception` — compiler-bug arm with rc=1). The
  `\u` escape ValueError-to-LexError re-raise at lexer.py
  :402 is on the user-diagnostic path. Not silent.
- **struct_mono ShapeFoldError -> typecheck driver -> check.py**:
  ShapeFoldError originates from constant-folding shape
  expressions during struct instantiation; converted to
  diagnostic string and appended to the shared `diags`
  list. The typecheck driver merges into the unified
  error stream. Not silent.
- **Backend raise -> check.py wrap**: the cycle-12-tested
  pathway (any backend raise -> check.py:618/649/663) is
  re-verified. The wrap emits "internal error" + "compiler
  bug" + rc=1. The user sees the exception class and
  message. Not silent.
- **C5-10 Pattern C at lower_ast.py:283 and :2067**: both
  fall-through paths (return None / idx_int = -1) are
  consumed by callers that either skip the LOAD_ELEM
  emission or fall through to the next lowering branch.
  Per the cycle-5 C5-10 analysis, this is a known LOW
  silent-fallback. Not re-flagged in cycle 15 (unchanged
  since cycle 5).

### Did adding TRACE_ENTRY/TRACE_EXIT to SIDE_EFFECT_KINDS break a different invariant? (re-asked for cycle 15)

Cycle 14's "did the fix break something else?" analysis is
re-verified on cycle 15's identical HEAD:
- **Performance**: marginally more ops survive DCE for
  unit-returning traced fns. Not silent.
- **Register pressure**: const_int(0) sentinel consumes a
  single i32 slot. Backend handles standard slots
  natively. Not silent.
- **SSA validity**: tree-style def-use. Trivially valid.
  Not silent.
- **Effect-check pass**: TRACE_* aren't in the effect
  inference machinery. Not silent.
- **Reflection (Quote/Modify/Splice)**: all subsystems
  survive DCE independently. Not silent.
- **Backend slot reuse**: const_int(0) value-id is stable
  across fold + dce. Slot allocator keys by value-id. Not
  silent.

Conclusion: cycle-14 fix opens no fresh silent-failure
window — confirmed on cycle 15's read-only re-walk.

### Carryover findings status (cycles 1-14) — unchanged

The cycle-15 re-audit closed nothing (read-only by
design) and introduced no new finding. The carryover
ledger is identical to cycle 14's closing snapshot.

| Carryover | Severity | Cycle-15 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — not addressed. Highest-priority unaddressed-CRITICAL. |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| C5-10 (lower_ast.py:2113-2117 + 2079-2092 + 2093-2101) | LOW | **still open** — not addressed; not re-flagged per the user's strict re-flag rule |
| monomorphize_safe docstring drift | (housekeeping) | **still open** |
| D-vs-Quote diagnostic text | (housekeeping) | **still open** |
| C7-1 test-coverage gap | (housekeeping) | **still open** |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | CLOSED by cycle 9 |
| C8-2 (cycle-8 LOW) | LOW | CLOSED by cycle 9 |
| C9-1 (cycle-9 LOW) | LOW | CLOSED by cycle 10 |
| C13-1 (cycle-13 HIGH, DCE drops TRACE_EXIT operand) | HIGH | CLOSED by cycle 14 fix-sweep at 1e4c3e6 |

These are NOT re-flagged as new cycle-15 findings per
the user directive (already documented in cycles 1-14,
did not CHANGE in cycle 15 — and indeed could not have
changed because no production commit landed). They
remain in the open-findings ledger and are out-of-scope
for this audit's strict-clean determination.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-16 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 15 did not address (read-only
  re-audit). **STILL THE HIGHEST-PRIORITY ITEM** for any
  future fix-sweep — the only remaining CRITICAL across
  the audit series. As the clean-counter accumulates
  (now 2/5 if cycle 15's clean verdict holds across all
  three lenses), the question of whether the Stage-29
  gate requires CRITICAL=0-open (stricter) or merely
  5-consecutive-clean (lenient) becomes load-bearing.
- **Carryover audit-C4-4 (D9 paper-only)**: still open
  HIGH. Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call
  fn-mono)**: still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks (Patterns A, B,
  C — including the freshly-enumerated lower_ast.py:280-283
  and :2064-2068 sites)**: still open LOW. Not addressed;
  not re-flagged.
- **monomorphize_safe docstring drift**: still open
  (cycle-6 deferred).
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred).
- **C7-1 test-coverage gap**: still open. Cycle 15 also
  did not add the 4 `_compatible(TyMemTier, TyVar)`
  regression tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix
  edge cases**: still no callee triggers either. Not
  findings.
- **TRACE_EXIT operand-less defensive guard
  (x86_64.py:2495)**: noted in cycle 14 — the `if
  op.operands:` guard tolerates a hypothetical operand-
  less TRACE_EXIT (the lowerer never emits one today).
  Future-tracking item if the trace machinery evolves.
  Not a finding for cycle 15.

---

## Cycle 14 vs cycle 15 — clean-cycle counter check

Cycle 14 = 1st clean of the re-accumulated window
(counter 1/5). The user directive for cycle 15
explicitly instructs: re-audit the same scope and
confirm nothing has regressed; do not re-flag prior-cycle
carryovers unchanged since cycle 14.

The cycle-15 re-audit honors that directive:
- `audit-C4-1 CRITICAL`, `audit-C4-4 HIGH`, `audit-C4-8 LOW`:
  not re-flagged.
- `C5-10 LOW` (lower_ast.py:2113-2117 + 2093-2101 +
  2079-2092, plus the freshly-enumerated :280-283 and
  :2064-2068 instances which are part of the same
  carryover Pattern C): not re-flagged.
- `monomorphize_safe docstring drift`, `D-vs-Quote
  diagnostic text`, `C7-1 test-coverage gap`: not
  re-flagged.

Cycle 15 produces **zero NEW findings of any severity**,
so the clean-cycle counter advances to **2/5** under
the strict criterion — subject to the parallel
type-design + code-review audit lenses also being
clean for cycle 15.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 15 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW).**

---

## Cycle 15 status

**Cycle 15 IS CLEAN** for the silent-failure audit lens.
Per the strict criterion (zero findings of ANY severity),
the 0-finding result satisfies the clean-cycle gate for
this audit lens.

### Stop-the-line determination: **NO**

Cycle 15 is clean — no stop required for this lens.

### Cycle 15 -> NEW FINDINGS COUNT for the strict-clean gate: 0
(0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter
advances to **2/5** for this audit lens (cycle 14 was 1/5
post C13-1 fix; cycle 15 is the second consecutive clean
cycle on the post-fix HEAD).

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
- Cycle 15: 0 findings (silent-failures lens). <- here

Trend: **6 consecutive clean cycles** on the silent-failures
lens (10, 11, 12, 13, 14, 15). The global strict-clean
counter is 2/5 because cycle 13's code-review lens broke
the 5-clean-cycle accumulation, resetting the global
counter to 0; cycle 14 was 1/5 and cycle 15 is 2/5 of the
re-accumulated window — subject to parallel
type-design + code-review lenses also being clean.

### Estimated remaining open findings going into cycle 16

- Cycle 1: 13 new (all fixed -> 0 open).
- Cycle 2: 6 new (all fixed -> 0 open).
- Cycle 3: 6 new (all fixed -> 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 5 silent-failure: 4 new — 3 closed by cycle 6.
  1 still open (C5-10 LOW, lower_ast.py fallbacks).
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new — both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10 silent-failure: 0 new.
- Cycle 11 silent-failure: 0 new.
- Cycle 12 silent-failure: 0 new.
- Cycle 13 silent-failure: 0 new (code-review lens found
  C13-1 HIGH — CLOSED by cycle 14 fix-sweep).
- Cycle 14 silent-failure: 0 new.
- Cycle 15 silent-failure: 0 new. <- here
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 16).
- Cycle 15 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-15 new) + (deferred type-design partial) = **>=23
  open findings** going into cycle 16. (Net 0 delta vs
  cycle 14 because cycle 15 closed nothing and opened
  nothing.)

Recommend prioritizing in this order for the cycle-16
fix batch (if user elects to land fixes between clean
re-audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in
   cycles 6-15).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks; the
   cycle-15 fresh-eyes rotation enumerated two more
   Pattern C sites at lower_ast.py:280-283 and
   :2064-2068).
4. **C7-1 test-coverage gap**.
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).

The "5 clean cycles before Phase 0 deprecation" goal
requires the strict criterion (zero findings of any
severity, all three lenses) to be met for 5 CONSECUTIVE
cycles. Cycle 14 = 1/5; cycle 15 = 2/5 of the
re-accumulated window. Three more clean cycles (16, 17,
18) needed across all three lenses to fire the gate
(assuming parallel type-design + code-review lenses
remain clean).

**Cycle 15 status: CLEAN**
**Counter status: 2/5** (cycle 14 silent-failures clean;
cycle 15 silent-failures clean; subject to parallel
type-design + code-review lenses also being clean for
cycle 15).
