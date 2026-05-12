# Stage 28.9 cycle 67 — type-design adversarial audit

HEAD: `b367ff3` (cycle-66 fix-sweep)
Date: 2026-05-12
Pass: 6th adversarial pass since cycle-58
Mode: STRICT READ-ONLY. No edits made to any source file. One write only: this report.
Criterion: 0 findings at confidence >= 75% = PASS.

## Verdict

**FAIL** — 1 finding at confidence >= 75%.

## Findings

### C67-1 — flatten_modules._flatten_one applies intra-mod aliases to nested-mod items, leaking parent-scope name resolution

- Location: `helixc/frontend/flatten_modules.py:165-171` (cycle-66 fix block)
- Confidence: HIGH (~90)
- Severity: HIGH (silent miscompile — cross-scope name capture)

**Mechanism.**
Within `_flatten_one(mb, prefix, new_items)`:

```python
direct_lifts_start = len(new_items)
n = 0
for sub in mb.items:
    if isinstance(sub, A.ModBlock):
        n += _flatten_one(sub, prefix=base, new_items=new_items)   # nested recursion appends to new_items
    elif isinstance(sub, A.FnDecl):
        new_items.append(A.FnDecl(... new_name = base + "__" + sub.name ...))
    ...
if intra_mod_aliases:
    for i in range(direct_lifts_start, len(new_items)):     # <-- includes nested-recursion appends
        it = new_items[i]
        if isinstance(it, A.FnDecl) and not it.is_extern:
            it.body = _rewrite_expr(it.body, intra_mod_aliases)
        elif isinstance(it, A.ConstDecl):
            it.value = _rewrite_expr(it.value, intra_mod_aliases)
```

The block-comment at lines 102-104 explicitly claims "nested ModBlock recursion adds to new_items too but for a different intra-mod scope — we don't rewrite those here." The implementation contradicts that claim: the slice `range(direct_lifts_start, len(new_items))` runs from the snapshot taken BEFORE the loop to the end of the loop, which covers every item appended during this call — including items appended by recursive `_flatten_one(sub_modblock, ...)` calls.

**Triggering input (simplest reproducer that surfaces the silent bug).**

```helix
mod outer {
    mod inner {
        fn baz() -> i32 { bar() }    // intra-inner-scope: `bar` is NOT a sibling of baz
    }
    fn bar() -> i32 { 7 }
}

fn main() -> i32 { outer::inner::baz() }
```

Expected language semantics (matching Rust-style module scoping the parser/typer assume elsewhere): the call `bar()` inside `inner::baz` should be an *unresolved-name* error — `bar` lives in `outer`, not in `inner`, and there is no `use super::bar;`.

Actual post-cycle-66 behavior:
1. The inner recursion `_flatten_one(inner, prefix="outer", ...)` appends `outer__inner__baz` to `new_items` and rewrites its body with `intra_mod_aliases_inner = {"baz": "outer__inner__baz"}`. The call `bar()` is not in that map, so it survives as `Name("bar")`.
2. Control returns to the outer `_flatten_one(outer, "", ...)`. Its `intra_mod_aliases = {"inner": "outer__inner", "bar": "outer__bar"}`. Its post-loop rewrite block iterates `range(direct_lifts_start=0, len(new_items))`, picking up `outer__inner__baz` (a nested lift, not a direct sibling) and rewriting its body with the OUTER aliases. The `bar()` callee matches `intra_mod_aliases["bar"] = "outer__bar"`.
3. Result: `outer__inner__baz` silently links to `outer__bar`, producing exit code 7 instead of the expected diagnostic.

**Second symptom — collision between cousin names.** If `outer` defines a sibling `bar` and `inner` ALSO defines a sibling `bar` (legal — distinct scopes), the outer rewrite overwrites inner-scope `bar()` references inside `outer__inner__baz`. The inner pass already rewrote `bar()` to `outer__inner__bar`; the outer pass then re-rewrites `outer__inner__bar` callees — actually, here outer aliases key on `"bar"` literal, and the inner pass already replaced it, so the outer pass *would not* re-fire. BUT in the inverse case (inner has only one fn `bar`, outer has fn `bar` and the inner `bar` body uses an outer-scope name not present in inner aliases), the outer pass captures it. The double-rewrite surface is non-trivial; the simpler reproducer above is sufficient to demonstrate the defect.

**Coverage gap.** `helixc/tests/test_codegen.py::test_module_nested_blocks` exercises only `outer::inner::f()` called from `main` with no intra-mod sibling calls anywhere. There is no test that places a name in inner scope that aliases a name in outer scope. The cycle-66 commit added the alias-rewrite logic but no nested-mod test covering the parent-bleed surface — the new code-path is reachable by trivial input yet untested.

**Why this is a type-design defect not a parse-time one.**
The walker is the scope-resolution boundary between source-form modules and the flattened name space the lowerer/monomorphizer assume. Conflating "items appended during this stack frame" with "items lifted from this mod's direct items" is a scope-window error: the type of the alias map (`dict[str, str]`) implicitly carries a scope, and the for-loop slice is the only thing enforcing that scope. The slice is wrong, so the type's implicit scope is violated.

**Fix sketch (NOT applied — read-only audit).**
Track the post-loop range using a sibling-only list: collect indices appended by `mb.items` direct branches (not by recursive ModBlock calls). One approach: snapshot `len(new_items)` BEFORE each `sub` and only append `(idx,)` to a `direct_indices` list for the FnDecl/ConstDecl branches, then iterate `direct_indices` in the rewrite pass. Or: change the recursion to return its own appended-index range and have the caller exclude it.

## Areas audited with no >=75%-conf findings

- `helixc/ir/lower_ast.py` UnsafeBlock path (line 1765): pass-through to `_lower_block`; no type information lost; matches the documented capability-boundary discipline.
- `helixc/ir/lower_ast.py` Match path (line 1902): hard-fails if a Match reaches the lowerer, with a precise diagnostic pointing at `match_lower`. Defensive; correct.
- `helixc/ir/lower_ast.py` Range path (line 1933): returns None outside For — phase-0 limitation, not a type bug; consistent with cycle-58/59 discipline.
- `helixc/backend/x86_64.py` slot allocator (lines 878-1017): every TIR value gets 8 bytes; float/int classification respects `isize`/`usize` (C18-1 already addressed). No new register-class invariant defect.

## No-edit attestation

This audit performed only Read, Grep, Glob, and Bash. No source file was modified. The only Write was this report at `docs/audit-stage28-9-cycle67-type-design.md`.
