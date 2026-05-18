# Stage 53 Inc 1 — Implementation Blueprint

**Date:** 2026-05-17
**Status:** Ready to execute (pending Stage 52 closure)
**Source:** code-architect agent during Stage 52 gate-8 wait
**Estimated complexity:** 2 increments (Inc 1 here; Inc 2 = Assign-stmt verification)

## Goal

Close the LAST remaining modal-launder bypass: helper-function
indirection. Reproducer (currently silent miscompile):

```helix
fn launder(x: i32) -> Known<i32> { into_known(x) }
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let k: Known<i32> = launder(r);   // SILENT today
    from_known(k)
}
```

## Implementation diff

Four surgical edits to
`C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`.

### Change 1 — dict declaration in `__init__` (after line 733)

```python
# Stage 53 Inc 1: map user-defined function names → the modal kind
# of their declared return type (e.g. 'known', 'uncertain').
# Read-only after Pass 1 (_register_fn). Never cleared per-fn because
# it is purely derived from fn-decl sig and is global to the program.
self._fn_modal_return_kind: dict[str, str] = {}
```

### Change 2 — clear in `check()` (after line 816)

```python
# Stage 53 Inc 1: parallel clear for re-entrancy safety (LSP/REPL).
self._fn_modal_return_kind = {}
```

### Change 3 — populate in `_register_fn` (after line 1262)

```python
# Stage 53 Inc 1: if the declared return type is a modal wrapper,
# record the kind so _modal_origin_of_expr can propagate taint
# through user-defined helper functions.
if isinstance(sig.ret, TyModal):
    self._fn_modal_return_kind[fn.name] = sig.ret.kind
```

### Change 4 — extend `_modal_origin_of_expr` (~line 2455)

Replace body with:

```python
if isinstance(expr, A.Name):
    return self._modal_origin_provenance.get(expr.name)
if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
    callee = expr.callee.name
    # Stage 52 original: builtin modal eliminators.
    if callee in _MODAL_ELIM_TO_KIND:
        return _MODAL_ELIM_TO_KIND[callee]
    # Stage 53 Inc 1: user-defined helpers whose declared return
    # type is a modal wrapper. Closes the helper-fn indirection
    # laundering vector.
    if callee in self._fn_modal_return_kind:
        return self._fn_modal_return_kind[callee]
return None
```

## Regression tests (5)

### T1 — Positive: reproducer fires

Same as the Goal section's reproducer; assert launder fires
at the `let k: Known<i32> = launder(r);` line.

### T2 — Positive: 2-hop chain

```helix
fn step1(x: i32) -> Uncertain<i32> { into_uncertain(x) }
fn step2(x: i32) -> Known<i32> { into_known(x) }
fn main() -> i32 {
    let u: Uncertain<i32> = step1(42);
    let r: i32 = from_uncertain(u);
    let k: Known<i32> = step2(r);     // fires at step2(r)
    from_known(k)
}
```

### T3 — Negative: same-kind helper with untainted argument

```helix
fn wrap_known(x: i32) -> Known<i32> { into_known(x) }
fn main() -> i32 {
    let k: Known<i32> = wrap_known(99);   // no error
    from_known(k)
}
```

### T4 — Negative: helper takes Known, returns Known (pass-through)

```helix
fn rewrap(k: Known<i32>) -> Known<i32> { k }
fn main() -> i32 {
    let k1: Known<i32> = into_known(5);
    let k2: Known<i32> = rewrap(k1);   // no error
    from_known(k2)
}
```

### T5 — Edge case: mutual recursion across modal-typed fns

```helix
fn even_wrap(x: i32) -> Known<i32> { ... odd_wrap(x - 1) ... }
fn odd_wrap(x: i32) -> Known<i32>  { ... even_wrap(x - 1) ... }
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(3);
    let r: i32 = from_uncertain(u);
    let k: Known<i32> = even_wrap(r);   // fires
    from_known(k)
}
```

Mutual recursion is safe because Pass 1 (`_register_fn`) reads
signatures only — bodies are checked in Pass 2, so both fns are
in `_fn_modal_return_kind` before any body is checked.

## Cascading-defect risk areas (gate-1 audit probes)

1. **Inner block let-shadow restore** — gate-3 NEW-HIGH-1 territory.
   `let k = launder(r)` inside an inner block correctly scope-pops
   `k` at block exit; new taint source doesn't change scope discipline.

2. **Assign-stmt parallel-union** — gate-3 NEW-HIGH-4 territory.
   Branch A `x = launder(r)` (installs), branch B `x = 0` (clears).
   Existing union logic merges via `_modal_origin_provenance` keys,
   doesn't distinguish how taint was installed; should work.

3. **PatBind when scrutinee is `Call(user_fn, ...)`** — gate-6
   CRITICAL-1 territory. Stage 52 already handles this via the
   unified helper; new path feeds the helper, so PatBind taint
   propagates automatically.

4. **`_shadowed_builtin_names` interaction** — if user defines
   `fn from_uncertain(...)`, the shadow error fires AND the
   user-fn is registered. The helper checks `_MODAL_ELIM_TO_KIND`
   FIRST, so for shadowed builtin names the builtin wins — no
   false negative possible. (User must fix the shadow error
   regardless.)

## What Inc 2 will add

The Stage 53 Inc 2 increment was anticipated but turns out to be
trivial: the Assign-stmt path at `_check_expr`'s A.Assign branch
already calls `_modal_origin_of_expr` (per the gate-6 unified
helper), so it gets Stage 53 coverage automatically. Inc 2 is
just verification + adding any missing regression tests for the
Assign-via-user-fn-call path.

## How to verify

After applying:

```
python -m pytest helixc/tests/test_stage40_modal.py -k "stage53" --tb=line
python scripts/selfhost_cascade.py
```

Expected: 5 new tests pass; cascade unchanged (sha=a6f1ee44).

## Lineage

- Stage 40 closure gate-1 H1 documented helper-fn indirection
  as Phase-0 known limitation, deferred to "a future
  taint-tracking pass" (originally Stage 52, refined to Stage 53
  during Stage 52 gate-2 closure scope split).
- Stage 52 gate-6 introduced `_modal_origin_of_expr` as the
  unified taint-source helper. This was the architectural prep
  that makes Stage 53 a 4-line change instead of touching
  3 install sites.
- Code-explorer + code-architect agent collaboration during
  Stage 52 gate-7/8 wait windows produced this exact blueprint
  with zero idle time.
