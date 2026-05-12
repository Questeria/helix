# Stage 28.9 cycle 67 — Code review audit

**Date**: 2026-05-12
**HEAD**: `b367ff3` (cycle-66 fix-sweep: cycle-65 findings — pipeline order + intra-mod calls)
**Lens**: code review (6th adversarial pass since cycle-58)
**Strict criterion**: ZERO findings of ANY severity at confidence >= 75%.

---

## Result: FAIL

**2 findings at confidence >= 75%.**

| Severity | Count |
|---|---|
| HIGH    | 1 |
| MEDIUM  | 1 |
| LOW     | 0 |
| **>=75% total** | **2** |

Both findings carry forward unchanged from the cycle-65 code-review (C65-1 / C65-2) — cycle-66 was a fix-sweep that closed the cycle-65 SILENT-FAILURE and TYPE-DESIGN audits (intra-mod call rewriting + modules-vs-impls order) but did NOT touch the cycle-65 CODE-REVIEW audit's two findings. The cycle-66 commit body explicitly lists only the silent-failure CN-1 and type-design CN-1 lineage; C65-1 + C65-2 are not mentioned at all.

The "Prior findings C1-C66 already addressed — do not re-flag" rule does not apply to these two because they were *filed* in cycle 65 but *not addressed* in cycle 66.

Below-threshold (<75%): one observation (B67-1, conf 65) on the intra-mod alias-rewrite slice; surfaced in the cycle-67 TYPE-DESIGN audit (`audit-stage28-9-cycle67-type-design.md::C67-1` HIGH conf ~90) so not re-counted here to avoid double-billing across audit lenses.

---

## Findings

### C67-CR-1 — `typecheck` STILL runs before `flatten_modules` in `helixc/check.py`; mod-nested fns silently escape type checking (HIGH, conf 92)

**Status**: live at HEAD `b367ff3`. C65-1 verbatim — un-addressed by cycle-66.

`helixc/check.py:403` invokes `typecheck(prog)`. `helixc/check.py:449` invokes `flatten_modules(prog)`. The cycle-66 fix-sweep reshuffled `flatten_modules` to BEFORE `flatten_impls` (lines 430-453) but left typecheck stranded at its pre-flatten position.

`helixc/frontend/typecheck.py:404, 411, 419, 862` all iterate `self.prog.items` directly. The class body contains zero references to `ModBlock` or `ImplBlock`. So `fn` items inside `mod m { ... }` blocks are silently skipped during `_register_fn` (line 414) and `_check_fn` (line 422) — the diagnostic surface for type errors, arity mismatches, return-type mismatches, generic-substitution failures, struct-field mismatches, and effect violations is bypassed for every mod-nested fn.

**Repro at HEAD `b367ff3`** (verified during this audit):

Source `/tmp/check67_test.hx`:
```hx
mod inner {
    fn bad(x: i32) -> i32 { x + true }
}
fn main() -> i32 { 0 }
```

Command + observed output:
```
$ python -m helixc.check --check-only /tmp/check67_test.hx
-- helixc-check: ...
   parse:    OK  (1 fns, 2 items)
   typecheck: OK
   totality:  OK
-- clean (check-only)
```

`i32 + bool` inside `mod inner::bad` is a textbook type error. The surface tool returns rc=0.

**Backend asymmetry** (the C65-1 / cycle-63 CN-A defect class):
- `helixc/backend/x86_64.py:3104` runs `flatten_modules`
- `helixc/backend/x86_64.py:3107` runs `flatten_impls`
- `helixc/backend/x86_64.py:3146` runs `typecheck`

So `python -m helixc.backend.x86_64 ...` would correctly type-check mod-nested fns, but `python -m helixc.check ...` (the canonical surface tool documented in `check.py:39` example list) does not. The exact same shape as cycle-63 CN-A for `emit_warnings`/`flatten_modules` — a pipeline contract the docstring narrative assumes both drivers obey but only the backend does.

**Severity rationale**: HIGH. Wider blast radius than any individual cycle-58..66 walker-drift finding. Typecheck is the foundational static gate; silently passing arbitrary mod-nested type errors in the user-facing tool defeats the entire purpose of `helixc check`.

**Fix sketch (NOT applied — read-only audit)**: move the `typecheck(prog)` call from line 403 to AFTER the `flatten_modules` + `flatten_impls` block (after line 471). Verify that any test using mod-nested fns with `--check-only` (none observed at this revision) plus the new pipeline order produces the expected diagnostics.

---

### C67-CR-2 — `autotune.collect_autotuned_fns` STILL hand-rolls top-level-only `prog.items` walk; cycle-60 walker-drift discipline not extended (MEDIUM, conf 80)

**Status**: live at HEAD `b367ff3`. C65-2 verbatim — un-addressed by cycle-66.

`helixc/frontend/autotune.py:228-231`:
```python
def collect_autotuned_fns(prog: A.Program) -> list[A.FnDecl]:
    """All top-level fn decls with @autotune."""
    return [it for it in prog.items
            if isinstance(it, A.FnDecl) and has_autotune(it)]
```

Compare to the cycle-60 C59-1 fix that routed `panic_pass.collect_panics`, `panic_pass.validate_panic_args`, `panic_pass.find_unwind_attrs`, `unsafe_pass.find_unsafe_blocks`, `unsafe_pass.find_raw_ptr_ops`, `trace_pass.traced_fn_names`, and `trace_pass.validate_trace_attrs` through `iter_fn_decls` (`from .ast_walker import iter_fn_decls`). `autotune.collect_autotuned_fns` is the only Item-level pass in the cycle-60 cohort that was missed and remains the lone hand-roll.

**Repro at HEAD `b367ff3`** (verified during this audit):
```python
from helixc.frontend.parser import parse
from helixc.frontend.autotune import collect_autotuned_fns, validate_autotune_prog
prog = parse('mod inner { @kernel @autotune(BS: [16, 32, 64]) fn k() -> i32 { 0 } }')
collect_autotuned_fns(prog)   # -> []
validate_autotune_prog(prog)  # -> []
```

`@autotune` declarations inside `mod` blocks are invisible to the validation pass when invoked pre-flatten. Production `check.py` runs `flatten_modules` (line 449) before `validate_autotune_prog` (line 575), so the gap is masked end-to-end through the canonical CLI; but direct-API callers (tests, REPL harnesses, future scripts, the bootstrap path probing autotune metadata) silently get empty results — identical to the production-masked-but-direct-API-exposed pattern that cycle-60 C59-1 fixed defensively for panic/trace/unsafe even though those were also production-masked.

**Docstring contract gap**: `panic_pass.collect_panics`, `unsafe_pass.find_unsafe_blocks`, etc. all carry the cycle-60 C59-1 comment marking `iter_fn_decls` recursion as policy. `autotune.collect_autotuned_fns` carries no such contract — its docstring (`"All top-level fn decls with @autotune."`) literally says "top-level," which contradicts how every sibling pass now behaves. `deprecated_pass.find_deprecated_decls` at `:79-110` resolves the same shape by documenting an explicit post-flatten contract; `autotune` does neither — neither recurses nor documents.

**Severity rationale**: MEDIUM. Production CLI is masked. The defect surfaces only on direct API use, which (per cycle-65 code-review table) is the same exposure shape that prompted the cycle-60 fix for the other three passes. Cross-pass discipline-asymmetry, not a user-observable miscompile via the documented entry point.

**Fix sketch (NOT applied — read-only audit)**:
```python
from .ast_walker import iter_fn_decls
def collect_autotuned_fns(prog: A.Program) -> list[A.FnDecl]:
    return [fn for fn in iter_fn_decls(prog) if has_autotune(fn)]
```
Mirror the docstring discipline of `panic_pass.collect_panics` (cycle-60 C59-1 fix comment).

---

## Areas audited with no >=75%-conf findings

- `helixc/frontend/grad_pass.py` Item-level walk (lines 81-85, 111-114, 120-122, 127-131, 136-138): hand-rolls `prog.items` iteration but BOTH production drivers run `grad_pass` strictly post-flatten (`check.py:642` is inside the emit branch after `flatten_modules` at line 449; `backend/x86_64.py:3134` after `flatten_modules` at line 3104). Cycle-65 audit table marks this "safe by construction." Confirmed re-reading at HEAD `b367ff3`.

- `helixc/frontend/deprecated_pass.py`: `_walk_items_for_fns` (lines 142-161) recurses into `ImplBlock.methods` and `ModBlock.items` explicitly (cycle-57 C57-5). `find_deprecated_decls` carries the explicit post-flatten contract docstring (lines 79-110). Both shapes correct.

- `helixc/frontend/totality.py` Item walker: cycle-58/60 fix; hand-rolled recursion but documented and tested (B65-3 sub-75 observation carried forward but stylistic only).

- `helixc/frontend/trace_pass.py`, `panic_pass.py`, `unsafe_pass.py`: all use `iter_fn_decls` (cycle-60 C59-1). Confirmed.

- `helixc/frontend/flatten_modules.py` cycle-66 fix block (lines 156-172): the intra-mod alias re-walk slice `range(direct_lifts_start, len(new_items))` includes items appended by recursive `_flatten_one` calls for nested ModBlocks. This contradicts the inline comment on lines 102-104 ("we don't rewrite those here") and surfaces as the cycle-67 TYPE-DESIGN audit's C67-1 (HIGH conf ~90). Not re-flagged here to avoid double-billing across audit lenses; cross-reference `docs/audit-stage28-9-cycle67-type-design.md::C67-1`.

- `helixc/frontend/flatten_modules.py` regression-test coverage (cycle-66 `test_c65_cn1_intra_mod_calls_rewritten` + `test_c65_cn1_totality_catches_mod_nested_recursion` in `test_deprecated.py:362-429`): two tests added. Both single-level (one `mod inner { ... }` block); no nested-mod scenario in the new tests. B67-1 observation (sub-75): nested-mod intra-call rewriting goes through the new code path on every program with nested mods, yet there is no targeted regression test covering it. Stylistic / future-proofing only; will be subsumed by the cycle-67 type-design C67-1 fix-sweep.

---

## Cross-file consistency check (delta from cycle 65)

Repeating the cycle-65 table for the two changed rows + new row:

| Module / function | Walker style | Production driver order | Direct-API safety | Status vs C65 |
|---|---|---|---|---|
| `typecheck.check` | top-level only | **PRE-flatten in check.py**, post-flatten in backend | **C67-CR-1 HIGH** | **unchanged from C65-1** |
| `autotune.collect_autotuned_fns` | top-level only | post-flatten in check.py (backend doesn't call) | **C67-CR-2 MED** | **unchanged from C65-2** |
| `flatten_modules._flatten_one` intra-mod rewrite slice | new code path | recursive ModBlock items leak parent-scope alias map | C67-1 HIGH (in type-design audit) | **new in cycle 66** |

All other rows (panic / unsafe / trace / deprecated / totality / grad_pass / hash_cons / struct_mono / flatten_impls / monomorphize / match_lower) carry forward from the cycle-65 table unchanged. No new walker-drift in any pass beyond the C67-1 cycle-66 self-introduction.

---

## Edits made

NONE. Strict read-only mode per the cycle-67 instructions. Tooling used: Read / Grep / Glob / Bash for inspection (including two read-only Python probes verifying the C67-CR-1 and C67-CR-2 repro inputs). The only Write was this report at `docs/audit-stage28-9-cycle67-codereview.md`. No source file was modified.

---

## Verdict

**FAIL — 2 findings at conf >= 75% (1 HIGH, 1 MEDIUM).**

Cycle-66 closed the cycle-65 silent-failure CN-1 (intra-mod call rewriting) and the cycle-65 type-design CN-1 (modules-vs-impls order parity with backend), but elided the cycle-65 CODE-REVIEW audit's two findings entirely — neither `typecheck`'s pre-flatten position nor `autotune.collect_autotuned_fns`'s top-level walk has changed at HEAD `b367ff3`. The fix-sweep also self-introduced a new HIGH (cycle-67 type-design C67-1) via the `direct_lifts_start` slice including nested-recursion appends. The clean-streak does not advance; cycle-68 should be a fix-sweep closing C67-CR-1, C67-CR-2, and C67-1.
