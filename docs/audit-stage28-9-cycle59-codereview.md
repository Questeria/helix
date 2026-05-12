# Audit Stage 28.9 cycle 59 — Code review

**Date**: 2026-05-11
**HEAD**: `722baf8` (cycle-58 fix-sweep follow-on: 5 walker-drift fixes + totality recursion)
**Lens**: code review (3rd adversarial pass on cycle-58 fix-sweep)
**Strict criterion**: ZERO findings of ANY severity at confidence >= 75%.

---

## Result: FAIL

**3 findings at confidence >= 75%.**

| Severity | Count |
|---|---|
| HIGH    | 1 |
| MEDIUM  | 2 |
| LOW     | 0 |
| **>=75% Total** | **3** |

Below-threshold (<75%): 2 (B59-1 comment drift `scan_items`/`_walk_items_for_fns` conf 70; B59-2 zero new dedicated tests for C57-1..C57-4 conf 72).

---

## Findings

### C59-1 — Walker-drift gap NOT closed in panic_pass / unsafe_pass / trace_pass (HIGH, conf 88)

**Scope**: cycle-58 closed the ModBlock/ImplBlock item-walker gap in `deprecated_pass.find_deprecation_call_sites` (C57-5) and `totality.check_totality` (C57-1 totality). Three sibling Item-level walkers in the same `helixc.frontend` package still iterate only `prog.items` filtered for top-level `A.FnDecl`:

- `panic_pass.collect_panics` (line 84) and `panic_pass.validate_panic_args` (line 125)
- `unsafe_pass.find_unsafe_blocks` (line 85) and `unsafe_pass.find_raw_ptr_ops` (line 148)
- `trace_pass` item-iteration (line 115) — same class

**Concrete repro** (verified in cycle-59 audit harness):

```hx
mod inner { fn x() -> i32 { panic("oops"); 0 } }
fn main() -> i32 { 0 }
```

Through the `helixc check` pipeline (which runs `flatten_impls` but NOT `flatten_modules` before these passes), `collect_panics(prog)` returns `[]` and `validate_panic_args(prog)` returns `[]`. The same pattern leaves panics inside `mod` blocks silently uncaught at the surface tool, identical UX impact as the original C57-5 finding for deprecated_pass.

`unsafe_pass.find_unsafe_blocks` reproducibly returns `[]` for the same ModBlock-wrapped fn. The backend pipeline (x86_64.py) does run `flatten_modules` early so the gap is masked end-to-end for emit, but `helixc check` users see no unsafe-pass diagnostics for mod-nested code.

**Defect class**: identical to C16-1 / C57-5 ("hand-rolled item walker missing ModBlock/ImplBlock recursion"). The cycle-58 fix-sweep claimed to close the gap "across all walkers" but stopped at deprecated_pass + totality. A new defensive helper analogous to `deprecated_pass._walk_items_for_fns` was not extended to its three sibling passes.

**Severity**: HIGH because it produces silent missing diagnostics in a user-facing pass invoked by `helixc check`.

---

### C59-2 — C57-5 fix is dead code in production pipelines for the ImplBlock branch (MEDIUM, conf 80)

`deprecated_pass._walk_items_for_fns` now recurses through `A.ImplBlock.methods`. But both production drivers (`helixc/check.py` and `helixc/backend/x86_64.py`) run `flatten_impls(prog)` BEFORE `deprecated_pass.emit_warnings(prog)`. After `flatten_impls`, every `ImplBlock.methods[i]` has been lifted to a top-level `FnDecl(name=Type__method)` and the ImplBlock removed. Cycle-59 audit verified empirically: post-flatten, the ImplBlock branch of `_walk_items_for_fns` is never entered when called from production drivers.

This is partial dead code — the branch is exercised only by direct test invocations of `find_deprecation_call_sites(parse(src))` that skip the flatten pass (none exist in the repo as of HEAD `722baf8`). The ModBlock branch IS live in `check.py` (which does not run `flatten_modules`). Same observation applies symmetrically to `totality.check_totality`'s ImplBlock branch — methods are lifted by flatten_impls before totality runs in either driver, so `test_c57_1_recursion_inside_impl_method_detected` exercises a code path that production callers don't reach.

Recommendation: either add a callable contract docstring stating "must be called pre-flatten" and add a direct test path that reproduces a user-visible bug, OR remove the ImplBlock branch and rely on flatten_impls being in the pipeline. Currently the code claims defense-in-depth that the test suite supports but no production path exercises.

---

### C59-3 — `find_deprecated_decls` not extended to ImplBlock/ModBlock — asymmetric with C57-5 fix (MEDIUM, conf 78)

The C57-5 commit message says: "deprecated_pass.find_deprecation_call_sites iterated only top-level FnDecl, skipping ImplBlock.methods and ModBlock.items. Deprecated functions called from impl methods produced no warning."

The fix added recursion for **call sites** but not for **declarations**. `find_deprecated_decls` (deprecated_pass.py line 79) still iterates only `prog.items` at the top level. Cycle-59 audit verified:

```hx
mod inner {
    @deprecated("old api")
    fn old_fn() -> i32 { 0 }
    fn caller() -> i32 { old_fn() }
}
```

Pre-flatten, `find_deprecated_decls(prog)` returns `{}` — the `@deprecated` decoration is invisible. `find_deprecation_call_sites` then has no `deps` to match and returns `[]`. The C57-5 fix's claim that mod-nested call sites are caught only holds when the deprecated decl is also at top level. In the production pipeline this is masked because `flatten_modules` (backend) and `flatten_impls` (check.py + backend) lift the decls before deprecated_pass runs. But the asymmetry between the new call-site walker and the unchanged decl walker is a comment-vs-code drift: the fix doesn't fully match its narrative.

---

## Below-threshold observations

- **B59-1** (conf 70): `totality.py` docstring (line 49) references the helper as `scan_items` ("Same item-walker gap as deprecated_pass C57-5 already closed via `scan_items`"). The actual helper is named `_walk_items_for_fns`. Mid-sweep rename not propagated. Documentation-only.
- **B59-2** (conf 72): C57-1 (flatten_impls UnsafeBlock+TileLit), C57-2 (flatten_modules UnsafeBlock+TileLit), C57-3 (pytree TyGeneric struct-ref), C57-4 (match_lower catchall) all landed with NO new dedicated regression tests. Only C57-1 (totality) added `test_c57_1_recursion_inside_mod_block_detected` and `test_c57_1_recursion_inside_impl_method_detected`. Future drift unprotected.
- **B59-3** (conf 60): `pytree._is_struct_ref` uses a local-scope `from .struct_mono import mangle_struct` import on every call inside `flatten_pytree` and `pytree_depth` and `_unflatten`. Comment justifies as "cheap insurance against future circular-import" but the actual call frequency on big pytrees could be noticeable. Move to module top with a TYPE_CHECKING / lazy guard if struct_mono ever needs pytree.

---

## Edits made

NONE. This audit was conducted in strict read-only mode per the cycle-59 instructions. No source files were modified; only this audit document was written.

---

## Verdict

**FAIL — 3 findings at conf >= 75% (1 HIGH, 2 MEDIUM).**

The cycle-58 fix-sweep closed the named drift cases but did NOT propagate the discipline to the three sibling Item-level walkers (panic_pass, unsafe_pass, trace_pass — C59-1 HIGH), and the ImplBlock branches added to the two patched walkers are dead-code through the production pipelines (C59-2 MED), and the C57-5 fix narrative claims symmetric coverage of decls + call sites but the decl walker is unchanged (C59-3 MED). Clean-streak does not advance; cycle-60 should be a fix-sweep closing C59-1..C59-3.
