# Audit Stage 28.9 cycle 67 — silent-failures scan

**HEAD**: `b367ff3df90b5f3fafbfa655a7eadd965b029b31`
**Branch**: main (Helix Phase-0 compiler)
**Date**: 2026-05-12
**Audit class**: silent-failures (6th adversarial pass at cycle 67)
**Mode**: STRICT READ-ONLY — no code edits; one Write call for this doc only.

## Verdict

**FAIL** — 4 findings at confidence >= 75%.

## Pipeline order observed at HEAD b367ff3

### `helixc/check.py` (surface CLI driver) — `_main_inner` order

| Step | Line | Pass |
|------|------|------|
| 1 | 380-389 | `parse` |
| 2 | 403 | **`typecheck(prog)`** |
| 3 | 423 | `monomorphize_structs(prog)` |
| 4 | 449 | `flatten_modules(prog)` |
| 5 | 467 | `flatten_impls(prog)` |
| 6 | 484 | `check_totality(prog)` |
| 7 | 513 | `emit_warnings` (`@deprecated` pass) |
| 8 | 527 | `validate_trace_attrs` |
| 9 | 543-544 | `validate_panic_args`, `validate_unwind` |
| 10 | 560 | `check_unsafe_ops` |
| 11 | 575 | `validate_autotune_prog` |
| 12 | 642 | `grad_pass(prog)` (only on emit path; never on `--check-only`) |
| 13 | 643 | `lower(prog)` |
| 14 | 668-678 | fdce / const_fold / cse / dce / `-O3` no-op |
| 15 | 710 | `effect_check_module(mod)` |

### `helixc/backend/x86_64.py` `__main__` driver order

| Step | Line | Pass |
|------|------|------|
| 1 | 3103 | `parse` |
| 2 | 3104 | `flatten_modules(prog)` |
| 3 | 3107 | `flatten_impls(prog)` |
| 4 | 3112 | `monomorphize_structs(prog)` |
| 5 | 3127 | `monomorphize_safe(prog)` (fn-level mono — NOT invoked by check.py at all) |
| 6 | 3134 | `grad_pass(prog)` |
| 7 | 3146 | **`typecheck(prog)`** |
| 8 | 3165 | `hash_cons(prog)` |
| 9 | 3173 | `check_totality(prog)` |
| 10 | 3183 | `lower(prog)` |

### Side-by-side delta

```
check.py:    parse > typecheck > struct_mono > flatten_mods > flatten_impls > totality > ... > grad_pass > lower
x86_64.py:   parse > flatten_mods > flatten_impls > struct_mono > fn_mono > grad_pass > typecheck > totality > lower
```

The cycle-66 fix-sweep ONLY reordered `flatten_modules` vs `flatten_impls` relative to each other (so flatten_modules now runs first, matching backend). It did NOT move `typecheck` or `monomorphize_structs` past the flatten passes, and it did NOT add `monomorphize_safe` or `grad_pass` to check.py.

---

## Findings

### C67-1 (HIGH, conf 95) — `typecheck` still runs PRE-flatten in `helixc/check.py`; the C65-1 finding was NOT addressed by the cycle-66 fix-sweep

**Location**: `helixc/check.py` line 403 (typecheck invocation) vs line 449/467 (flatten passes).

**Evidence**:
- `helixc/frontend/typecheck.py` `Checker.check` (lines 397-426) iterates `self.prog.items` filtering for `A.StructDecl`, `A.EnumDecl`, `A.FnDecl` exclusively. **Zero references** to `ModBlock` or `ImplBlock` exist in `typecheck.py` (verified by Grep returning "No matches found").
- `helixc/frontend/ast_nodes.py` lines 532-557 confirm `ModBlock` and `ImplBlock` are distinct `Item` subclasses (NOT subclasses of `FnDecl`/`StructDecl`).
- `helixc/frontend/parser.py` lines 199, 241, 252 confirm the parser EMITS top-level `ModBlock` and `ImplBlock` nodes; lifting only happens in the flatten passes.
- `helixc/frontend/flatten_modules.py` line 80 confirms `flatten_modules` mutates `prog.items` in place to lift mod-nested items to top level.
- At check.py line 403, `prog.items` still contains un-flattened `ModBlock` / `ImplBlock` Items, so typecheck silently skips every fn nested inside `mod foo { ... }` and every method inside `impl Foo { ... }`.
- The backend driver (`x86_64.py` line 3146) runs typecheck AFTER both flatten passes — correct ordering.

**Consequence**: Type errors inside any `mod foo { fn bar() { ... } }` or `impl Foo { fn method(self) { ... } }` are silently DROPPED by `helixc check foo.hx` but caught by `python -m helixc.backend.x86_64 foo.hx out.bin`. This produces the exact "surface clean / backend rejects" UX asymmetry that the cycle-63 CN-A and cycle-65 C65-1 audits flagged.

This finding is EXPLICITLY listed in the user prompt as deferred from cycle-65 (C65-1 code-review) with the question "is this still an issue, or did the cycle-66 reorder also move flatten passes earlier than typecheck?" — answer: **typecheck was NOT moved**; cycle-66 only reordered the two flatten passes relative to each other.

**Recommendation**: Move the `typecheck(prog)` call from line 403 to AFTER `flatten_impls(prog)` at line 471 (mirroring x86_64.py:3146). Verify diagnostic spans on mod-nested items still render correctly — the `Span` carried by each AST node is preserved through flatten_modules' rename (only `name` is rewritten; spans are untouched).

---

### C67-2 (HIGH, conf 88) — `monomorphize_structs` runs PRE-flatten in `helixc/check.py`; the deferred CN-2 struct_mono pre-flatten asymmetry remains

**Location**: `helixc/check.py` line 423 (struct_mono) vs line 449 (flatten_modules) vs `x86_64.py` line 3104 (flatten_modules) vs line 3112 (struct_mono).

**Evidence**:
- check.py order: struct_mono BEFORE flatten_modules.
- x86_64.py order: struct_mono AFTER flatten_modules (and after flatten_impls).
- `prog.items` at check.py line 423 still contains `ModBlock` items; generic `StructDecl` instances nested inside `mod foo { struct Bar<T> { ... } }` are NOT visible at top level.
- If `monomorphize_structs` iterates only `prog.items` looking for top-level `StructDecl` (the standard pattern across this codebase), then mod-nested generic structs are silently skipped — arity-mismatch diagnostics, mangled-name registration, and shape-fold side-effects all miss. The backend runs the same pass AFTER flatten, where mod-nested generic decls have been lifted to top level — so the backend catches what check.py misses.

This finding matches the "struct_mono pre-flatten asymmetry (CN-2 from silent-failures, deferred)" explicitly listed in the user prompt as a deferred cycle-65 finding.

**Recommendation**: Move `monomorphize_structs(prog)` from line 423 to after `flatten_impls(prog)` at line 471 (mirroring x86_64.py line 3112's post-flatten ordering).

---

### C67-3 (MEDIUM-HIGH, conf 82) — `helixc/check.py` does NOT invoke `monomorphize_safe` (fn-level monomorphization) at all

**Location**: `helixc/check.py` (no occurrence) vs `helixc/backend/x86_64.py` line 3127.

**Evidence**:
- `Grep monomorphize_safe` on check.py: zero hits.
- `Grep monomorphize_safe` on x86_64.py: line 3127 — invoked between struct_mono and grad_pass, with shape-fold-error trap 28801 emission as `error: fn-mono:` and pipeline abort on diagnostics.
- check.py only runs `monomorphize_structs` (parametric structs). Generic FUNCTIONS that need fn-level monomorphization are skipped in the surface CLI.

**Consequence**: A user-authored generic fn whose instantiation would trigger trap 28801 (shape-fold error) silently passes `helixc check foo.hx` and `helixc check --check-only foo.hx`, but fails immediately when the user invokes the backend driver to produce a binary. Even on `helixc check -o out.bin foo.hx`, the backend's `compile_module_to_elf` IS called (line 791) but at that point typecheck and totality have already declared "clean" — the user receives mixed messaging: stages 1-11 all PASS, then codegen aborts with a fn-mono trap.

**Recommendation**: Add `monomorphize_safe(prog)` invocation in check.py between struct_mono and grad_pass (after flatten passes, per C67-2 recommendation), with the same diagnostic-and-abort behavior as x86_64.py lines 3127-3133. Note this finding is severity-coupled to C67-1 and C67-2: if those are fixed by moving typecheck/struct_mono post-flatten, then adding fn-mono in the same neighborhood is a natural completion of the pipeline alignment.

---

### C67-4 (MEDIUM, conf 78) — `grad_pass` runs at different pipeline positions in check.py vs x86_64.py; `--check-only` users see no grad expansion at all

**Location**: `helixc/check.py` line 642 (inside `--emit-*/-o` branch only) vs `helixc/backend/x86_64.py` line 3134 (before typecheck).

**Evidence**:
- check.py line 642: `grad_pass(prog)` is inside the `if any(f in a.flags for f in ("--emit-ir", "--emit-asm", "--emit-ptx")) or a.output is not None:` branch (line 615-616). It runs ONLY when the user requests IR / asm / ptx / `-o` output.
- check.py line 610: `if "--check-only" in a.flags: return 0` — exits BEFORE the emit branch. So `helixc check --check-only foo.hx` never runs grad_pass.
- check.py default-no-flag invocation: also returns early from the emit branch since no `--emit-*` flag is set and `a.output is None`. So `helixc check foo.hx` (no flags) ALSO skips grad_pass.
- x86_64.py line 3134: `grad_pass(prog)` runs unconditionally BEFORE typecheck.
- x86_64.py lines 3138-3145 acknowledge in-comment: "NOTE: grad_pass internally invokes lower_matches(), which desugars match -> if/let chains. So typecheck sees the lowered form... Suppressing those would require either teaching typecheck about the lowered form, or splitting grad_pass so lower_matches runs after typecheck."

**Consequence (two-pronged)**:
1. `helixc check foo.hx` and `helixc check --check-only foo.hx` produce no grad-pass diagnostics. A program using `grad(loss)` may silently parse, typecheck, totality-check, etc., yet expansion errors (unknown loss symbol, malformed call shape) never surface until the user adds `--emit-ir` or `-o`.
2. Even when check.py DOES run grad_pass (emit path), typecheck has already run at line 403 — BEFORE grad_pass. Backend runs grad_pass BEFORE typecheck. So a `match` expression that grad_pass desugars produces different typecheck verdicts in the two drivers: backend's typecheck sees the lowered `if/let` chain (surfacing the "fake enum-variant has payload" warnings noted in x86_64.py comment 3140-3142); check.py's typecheck sees the original `match`. Diagnostic divergence between the two drivers for the same source file.

**Recommendation**: Move `grad_pass(prog)` invocation to BEFORE typecheck in check.py (after flatten passes), mirroring x86_64.py line 3134. Lazy-import to keep startup cost low for non-grad programs is fine (x86_64.py also imports it at the top of `__main__`, but check.py's local-import-on-demand pattern is preserved).

---

## Sub-threshold notes (conf <75)

- **N-1** (conf 65): `dce.py` `SIDE_EFFECT_KINDS` includes `REFLECT_HASH` (line 50). Comment notes "provides a stable testing handle that downstream code may reach via cell indexing." Liveness is correct (operands of side-effecting ops are seeded live at line 103-105), but the assumption that ALL REFLECT_HASH-cell readers go through MODIFY/SPLICE ops (which are also seeded as side-effecting at lines 41-42) is undocumented. If a future stage adds a non-MODIFY reader that consumes the cell-index integer as a plain LOAD_VAR target, the cell index could be dropped by DCE between REFLECT_HASH emission and the reader. No current code path exercises this; flagging for future-stage attention only.
- **N-2** (conf 60): `cse.py` line 65-66 marks `MAXIMUM`/`MINIMUM`/`POW` as pure. `POW` of negative base with fractional exponent is implementation-defined in IEEE 754 (NaN result). The audit comment at line 69-72 explicitly defers `EXP`/`LOG`/`SQRT` for the same reason but includes `POW`. Below threshold because the const-fold pass should refuse compile-time NaN via FoldError trap 17001, so a CSE merge of two POW ops with the same operands can't introduce new NaN — but the audit-trail asymmetry mirrors cycle-19's C19-1 concern.
- **N-3** (conf 60): `fdce.py` line 41 regex `_ID_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")` is rebuilt every call to `fdce_module`. Performance-only; not a correctness issue.

## Findings NOT re-flagged (already addressed C1-C66)

Per audit scope, prior findings already addressed are excluded. Specifically:
- Cycle-63 CN-1: check.py skipping `flatten_modules` (fixed cycle-64).
- Cycle-64 C64-Z: flatten order wrong (modules after impls) — fixed cycle-66.
- Cycle-66 fix: flatten_modules upgrade rewriting intra-mod self-calls — present in flatten_modules.py and verified in check.py comment 441-446.
- Cycle-28 C27-2/C27-3: effect_check classifier co-location — present and verified at check.py 632-635 + x86_64.py 3083-3086.
- Cycle-30 C29-R1/C29-1/C29-2: FoldError pre-prefix guard — present and verified at const_fold.py 72-87.
- Cycle-32 C31-3: `raise TypeError` instead of `assert` for FoldError body validation (survives `python -O`) — verified at const_fold.py 72-78.
- Cycle-13 C13-1: TRACE_ENTRY/TRACE_EXIT in SIDE_EFFECT_KINDS — verified at dce.py 79-80.

## Edits made

NONE. This audit was conducted in STRICT READ-ONLY mode. No source files were modified. The only Write call was for this audit document at `docs/audit-stage28-9-cycle67-silent-failures.md`.

## Files inspected

- `C:/Projects/Kovostov-Native/helixc/check.py` (full file)
- `C:/Projects/Kovostov-Native/helixc/backend/x86_64.py` (lines 1-100, 3060-3190)
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py` (lines 1-100, 395-446)
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_modules.py` (lines 1-80)
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_impls.py` (lines 1-60)
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py` (lines 525-559)
- `C:/Projects/Kovostov-Native/helixc/frontend/parser.py` (Grep ModBlock/ImplBlock production sites)
- `C:/Projects/Kovostov-Native/helixc/ir/passes/cse.py` (full file)
- `C:/Projects/Kovostov-Native/helixc/ir/passes/const_fold.py` (lines 1-100)
- `C:/Projects/Kovostov-Native/helixc/ir/passes/dce.py` (full file)
- `C:/Projects/Kovostov-Native/helixc/ir/passes/fdce.py` (full file)
- `C:/Projects/Kovostov-Native/docs/audit-stage28-9-cycle65-codereview.md` (corroborates C65-1)
- `C:/Projects/Kovostov-Native/docs/audit-stage28-9-cycle65-silent-failures.md` (header check)
