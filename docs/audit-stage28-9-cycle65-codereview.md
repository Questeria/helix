# Audit Stage 28.9 cycle 65 — Code review

**Date**: 2026-05-12
**HEAD**: `e7bd9c6` (cycle-64 fix-sweep: cycle-63 findings, pipeline contract)
**Lens**: code review (5th adversarial pass on cycles 56–64 walker-drift / pipeline-contract sweep)
**Strict criterion**: ZERO findings of ANY severity at confidence >= 75%.

---

## Result: FAIL

**2 findings at confidence >= 75%.**

| Severity | Count |
|---|---|
| HIGH    | 1 |
| MEDIUM  | 1 |
| LOW     | 0 |
| **>=75% Total** | **2** |

Below-threshold (<75%): 3 (B65-1 autotune walker-drift conf 72; B65-2 no regression tests for unsafe/trace mod-nested fixes conf 70; B65-3 totality + deprecated hand-roll Item walkers bypassing `iter_fn_decls` conf 65).

---

## Findings

### C65-1 — `typecheck` runs BEFORE `flatten_modules` in `helixc/check.py`; mod-nested fns silently escape type checking (HIGH, conf 92)

**Scope question explicitly invited this**: "helixc/check.py pipeline order: now that flatten_modules runs, what about typecheck — does it benefit too?"

Cycle 63/64 added `flatten_modules(prog)` to `helixc/check.py` (line 466) between `flatten_impls` (line 442) and the analysis passes (totality, deprecated, trace, panic, unsafe, autotune). The fix correctly aligned the surface tool with the backend on the panic/unsafe/trace/deprecated/totality side. However, `typecheck(prog)` is invoked at line 403 — BEFORE either flatten pass. `typecheck.py` itself contains zero references to `ModBlock` or `ImplBlock` and iterates only `self.prog.items` at the top level (lines 404, 411, 419, 862).

Concrete repro (verified at HEAD `e7bd9c6`):

```hx
mod inner {
    fn bad(x: i32) -> i32 { x + true }
}
fn main() -> i32 { 0 }
```

`python -m helixc.check --check-only <file>` returns rc=0 with output:

```
   parse:    OK  (1 fns, 2 items)
   typecheck: OK
   totality:  OK
-- clean (check-only)
```

The clear type error `i32 + bool` inside `mod inner::bad` is invisible. Without `--check-only` the same source still produces `typecheck: OK` (no `-o`, no `--emit-*`, no codegen path entered — typecheck remains the only gate that examined the program).

The backend driver (`helixc/backend/x86_64.py` line 3146) runs `typecheck` AFTER `flatten_modules` (line 3104) + `flatten_impls` (line 3107) + `monomorphize_structs` + `monomorphize_safe`, so the backend correctly type-checks mod-nested fns. This is the identical asymmetry that cycle-63 CN-A diagnosed for `emit_warnings`/`flatten_modules`: a pipeline contract the docstring narrative assumed held in both drivers, but only held in the backend.

**Defect class**: same as cycle-63 CN-A (HIGH conf 95). Pre-flatten `helixc check` produces silent-pass on user-visible errors that the backend would catch. The cycle-64 fix moved 5 analysis passes (deprecated, totality, trace, panic, unsafe) and 2 validation passes (autotune is at line 574) under the post-flatten contract but left typecheck stranded above the flatten step.

**Severity**: HIGH because it produces silent missing diagnostics in `helixc check` for the most foundational static check the compiler offers (type errors). Wider blast radius than cycle-63 CN-1 — typecheck catches dozens of error classes (arity, return-type, struct-field, generic-substitution) that all silently pass for mod-nested fns.

**Recommendation**: move `typecheck(prog)` invocation to AFTER both flatten passes in `helixc/check.py`. Verify no regression in typecheck-driven diagnostics whose error spans assume unflattened identifier shape (they shouldn't — the backend already does this).

---

### C65-2 — `autotune.collect_autotuned_fns` still hand-rolls top-level-only `prog.items` walk; identical defect class to fixed panic/unsafe/trace (MEDIUM, conf 80)

**Scope question explicitly invited this**: "now that ast_walker.iter_fn_decls is the shared helper, are there OTHER Item-level passes that still hand-roll their own `for it in prog.items` walk? Look at grad_pass, autotune, reflection."

`helixc/frontend/autotune.py` line 228-231:

```python
def collect_autotuned_fns(prog: A.Program) -> list[A.FnDecl]:
    """All top-level fn decls with @autotune."""
    return [it for it in prog.items
            if isinstance(it, A.FnDecl) and has_autotune(it)]
```

Concrete repro (verified at HEAD `e7bd9c6`):

```python
src = '''
mod inner {
    @kernel
    @autotune(BS: [16, 32, 64])
    fn k() -> i32 { 0 }
}
'''
prog = parse(src)
collect_autotuned_fns(prog)   # -> []
validate_autotune_prog(prog)  # -> []
```

The `@autotune` validation pass that cycle-60 wired into `check.py` (line 574) is masked in production because flatten_modules runs first at line 466. Direct-API callers (tests, future tools, REPL) get silently incorrect zero results — identical to the panic/unsafe/trace gap that cycle-60 C59-1 fixed by routing those passes through `iter_fn_decls`.

The other scope items: `grad_pass` (lines 81, 112, 120, 127, 136) hand-rolls top-level-only walks too — but `grad_pass` runs AFTER flatten in both drivers (check.py line 641 is inside the lower-branch; x86_64.py line 3134 runs after the flatten passes) so production is post-flatten by construction. No reflection module exists. So `autotune` is the only live offender among the scope's three named candidates.

**Severity**: MEDIUM because production drivers run flatten_modules first, masking the gap end-to-end. The gap is exercised only by direct API callers, but the cycle-60 walker-drift discipline says the Item-level walker should be canonical regardless of caller. Asymmetric with the cycle-60 fix for panic/unsafe/trace, which routed equally-post-flatten production paths through `iter_fn_decls` anyway as defensive infrastructure.

**Recommendation**: route `autotune.collect_autotuned_fns` through `iter_fn_decls` to match the cycle-60 cross-pass discipline.

---

## Below-threshold observations

- **B65-1 (conf 72)**: `autotune.collect_autotuned_fns` (see C65-2) is also dead-branch-equivalent in production drivers — same observation that cycle-59 C59-2 raised for the ImplBlock branch of `_walk_items_for_fns`. Documenting separately because the cycle-59 fix path chose "document and keep" over "remove and rely on flatten"; autotune has neither contract docstring nor recursion. Captured under C65-2 main finding.

- **B65-2 (conf 70)**: cycle-60 C59-1 fix touched `panic_pass.collect_panics`, `panic_pass.validate_panic_args`, `panic_pass.find_unwind_attrs`, `unsafe_pass.find_unsafe_blocks`, `unsafe_pass.find_raw_ptr_ops`, `trace_pass.traced_fn_names`, and `trace_pass.validate_trace_attrs`. Only one regression test landed (`test_c59_1_panic_in_mod_nested_fn_detected` at `test_deprecated.py:296` — testing panic only). Trace + unsafe mod-nested behavior empirically works (verified `find_unsafe_blocks` returns 1 and `traced_fn_names` returns `['t']` for a mod-nested case) but is uncovered. Recurring B59-2-class observation, still sub-75 in cycle 65.

- **B65-3 (conf 65)**: `iter_fn_decls` docstring claims it "centralises the walker-drift discipline so Item-level passes share the same item-walk surface — a future Item subclass that holds FnDecls forces an explicit dispatch decision in ONE place instead of N." Two cycle-58/60 walker fixes — `totality.collect_items` (lines 54-66) and `deprecated_pass._walk_items_for_fns` (lines 142-161) — hand-roll their own equivalent recursion instead of delegating to `iter_fn_decls`. If `prog.items` ever grows a new Item subclass (e.g., `TraitDecl` containing default-method `FnDecl`s) the centralisation claim would NOT hold — those two walkers must be hand-edited. Stylistic / future-proofing only at conf 65.

- **B65-4 (conf 50)**: `totality.py` docstring line 49 still references the helper as `scan_items` ("Same item-walker gap as deprecated_pass C57-5 already closed via `scan_items`"). Actual helper is named `_walk_items_for_fns`. This is B59-1 carried forward unchanged through cycles 60–64. Documentation-only, low confidence on impact.

- **B65-5 (conf 40)**: `ast_hash.py` line 392 contains `TODO follow-on cycle to recurse into GenericParam and WhereClause shapes; deferred because their hash surface is comparatively small`. The TODO predates 2026-04 freshness threshold... actually it was added 2026-05-04 (cycle commit `8bc962a`), so doesn't trip the scope rule, but is worth noting as the only TODO/FIXME/XXX/HACK in `helixc/frontend/` and the only one without a tracking ID. Out of scope per the freshness criterion.

---

## Cross-file consistency check (per scope)

Item-level passes that walk `prog.items`, post cycle-64:

| Module / function | Walker style | Production driver order | Direct-API safety |
|---|---|---|---|
| `panic_pass.{collect_panics, validate_panic_args, find_unwind_attrs}` | `iter_fn_decls` | post-flatten in check.py + backend | safe (recurses) |
| `unsafe_pass.{find_unsafe_blocks, find_raw_ptr_ops}` | `iter_fn_decls` | post-flatten | safe (recurses) |
| `trace_pass.{traced_fn_names, validate_trace_attrs}` | `iter_fn_decls` | post-flatten | safe (recurses) |
| `deprecated_pass.find_deprecated_decls` | top-level only | post-flatten (cycle-64) | documented contract |
| `deprecated_pass._walk_items_for_fns` | hand-rolled recursion | post-flatten | recurses (B65-3) |
| `totality.collect_items` | hand-rolled recursion | post-flatten (cycle-64) | recurses (B65-3) |
| `typecheck.check` | top-level only | **PRE-flatten in check.py**, post-flatten in backend | **C65-1 HIGH** |
| `autotune.collect_autotuned_fns` | top-level only | post-flatten | **C65-2 MED** |
| `grad_pass.{grad_pass, _has_grad_call, _resolve_let_aliases}` | top-level only | post-flatten (both drivers) | safe by construction (always post-flatten) |
| `hash_cons` | top-level only (line 86) | post-flatten if `--hash-cons` set (after autotune) | safe |
| `struct_mono.monomorphize_structs` | top-level + visits expr | pre-flatten | recurses internally |
| `flatten_impls.flatten_impls` | top-level | pre-flatten (it IS the flatten) | n/a |
| `flatten_modules.flatten_modules` | recursive | pre-flatten (it IS the flatten) | n/a |
| `monomorphize` | top-level + visits | between mono passes | n/a |
| `match_lower.lower_matches` | top-level + visits | inside grad_pass | safe (post-flatten via grad_pass position) |

`typecheck` and `autotune` are the only Item-level walkers that don't recurse into ModBlock/ImplBlock AND don't have the flatten passes preceding them in every driver. `typecheck` is the HIGH severity case because its ordering in check.py is genuinely pre-flatten; `autotune` is MED because production is masked but direct-API is exposed.

---

## Edits made

NONE. This audit was conducted in strict read-only mode per the cycle-65 instructions. No source files were modified; only this audit document was written. Inspection used Read / Grep / Glob / Bash (read-only python -c probes and pytest invocation).

---

## Verdict

**FAIL — 2 findings at conf >= 75% (1 HIGH, 1 MEDIUM).**

The cycle-64 fix-sweep correctly aligned `helixc check` with the backend on the post-flatten contract for 5 analysis passes (deprecated, totality, trace, panic, unsafe) and 1 validation pass (autotune at the driver level). However, `typecheck` was left at its pre-flatten position in `helixc/check.py` and silently passes mod-nested type errors that the backend rejects (C65-1 HIGH). And `autotune.collect_autotuned_fns` still hand-rolls a top-level-only walk that is masked end-to-end in production but exposed to direct-API callers (C65-2 MED). Both are the same defect class the cycle-58..64 sweep was supposed to extinguish, in scope items the audit instructions explicitly invited probing. Clean-streak does not advance; cycle-66 should be a fix-sweep closing C65-1..C65-2.
