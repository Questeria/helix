# Audit Stage 28.9 cycle 87 — Silent failures

**Scope:** HEAD `d8e5807` (Stage 28.9 cycle-86 fix landed: const_fold SHL/SHR
result-type bitwidth bound).

**Mode:** strict read-only — no Edit on source files.

**Rotation:** const_fold cycle-86 fix verification; struct_mono `mangle_struct`
injectivity; dce.py side-effect coverage.

**Out of scope (per audit charter):** Stage 28.10 and Stage 28.11 parallel
commits; deferred-known items (monomorphize `_mangle_ty` catch-all already
loud-fail per cycle-71 fix, hash_cons `_ast_equal`, struct_mono pre-flatten
ordering, autotune `collect_autotuned_fns`).

---

## Verification of cycle-86 fix (const_fold SHL/SHR bound)

cycle-86 changed the SHL/SHR const-fold bound from a hard-coded `[0, 63]` to
`_INT_BITS.get(result_ty.name, 64)`-derived. Two questions:

1. **Non-int result types (f32 / f64 / bool):** the SHL/SHR fold branch only
   activates when BOTH operand defs are `CONST_INT` (line 469-471). A
   `result_ty=f32 SHL` path is unreachable by construction — the frontend
   typechecker rejects SHL on float types before IR lowering, and even if it
   slipped through, the operand defs would not be `CONST_INT`. **Verified safe.**
   The `bool` case in `_INT_BITS` maps to 32 (per line 110 comment, IR reifies
   bool comparisons to i32), so a hypothetical `bool SHL` would use a 32-bit
   bound — defensible.

2. **Unknown int type name → fallback 64:** if `result_ty.name` is not in
   `_INT_BITS`, the `.get(..., 64)` returns 64. This matches the pre-fix
   behavior (hard-coded 63 → `[0, 64)`-equivalent) and is a strict
   non-regression. The only int type names this would miss are exotic
   future widths (i128, u128); when those are added, the cycle-86 author's
   intent (per-type-width bound) requires extending `_INT_BITS`, NOT a
   silent fallback. The 64 default is acceptable transitional behavior.

**Cycle-86 fix verifies clean.** No new findings.

---

## Rotation: helixc/frontend/struct_mono.py — `mangle_struct` injectivity

Examined the mangler at line 52-56. `mangle_struct` joins per-type-arg
mangles with single `_` separator, prefixed by `<StructName>__`. Reuses
`_mangle_ty` from `monomorphize.py`. `_mangle_ty` itself also uses `_` at
each composition level (e.g. `TyGeneric` arm at line 148-149 returns
`base + "_" + "_".join(args)`).

**Theoretical collision example:**

* `Pt<X<i32>>` (arity-1 generic with nested `X<i32>`) →
  `mangle_struct("Pt", [TyGeneric("X", [i32])])` →
  `"Pt__" + "X_i32"` = `"Pt__X_i32"`
* `Pt<X_i32>` (arity-1 generic with `X_i32` as a nominal TyName) →
  `mangle_struct("Pt", [TyName("X_i32")])` →
  `"Pt__" + "X_i32"` = `"Pt__X_i32"`

Reaching the collision requires the user to (a) declare a nominal struct
literally named `X_i32` (legal Helix identifier), (b) instantiate `Pt` once
with `X<i32>` and once with `X_i32`, in the same compilation unit.

`monomorphize_structs` line 376 uses a compound guard:

```python
if key not in rewrite_map and inst.name not in existing:
    rewrite_map[key] = inst.name
    mono_decls.append(inst)
    existing.add(inst.name)
```

`_ty_key` (line 244-304) produces distinct tuple keys for the two cases (one
is `("gen","X",...)`, one is `("name","X_i32")`) — so they pass the `key`
check. But the second arrival fails `inst.name not in existing` (cycle-3 C3-4
dedup-against-rerun guard) and is **silently dropped — no diagnostic appended
to `diags`, no entry in `rewrite_map`**. Downstream typecheck would resolve
both AST use-sites to the first-arrived `StructDecl`'s field layout. Silent
miscompile if the two structs had different field shapes.

**Confidence: ~70%.** The mangler IS non-injective on the construction
shown. Reachability gated on a user declaring an adversarially-named
companion struct (`X_i32`) AND using it alongside a parameterized nested
generic. This is unlikely in real code but legal at the surface syntax.
The fix is a single edit (length-prefix mangle, or use a non-identifier
separator like `__` between args and reserve `_` only within names) — the
sister modules `_ty_key` is already injective, so the bug is one-sided.

**Below the 75% threshold.** Not a hard finding for cycle-87.
Filed as soft observation; recommend opening as a deferred-known item or
cycle-88 rotation candidate. Not blocking.

---

## Rotation: helixc/ir/passes/dce.py — dead-code elimination correctness

Examined `SIDE_EFFECT_KINDS` (line 32-81) against the full `OpKind` enum at
`helixc/ir/tir.py:125-302`.

All side-effecting ops are listed: RETURN, BR, COND_BR, CALL, STORE_VAR,
STORE_ELEM, ALLOC_VAR, ALLOC_ARRAY, MODIFY, SPLICE, PRINT, QUOTE,
REFLECT_HASH, ARENA_PUSH, ARENA_SET, TILE_INDEX_STORE, FFI_CALL, TRAP,
TRACE_ENTRY, TRACE_EXIT.

Pure ops correctly omitted (the spread phase reaches their operands via
result-liveness): BITCAST, QUANTIZE, DEQUANTIZE, THREAD_IDX, TILE_INDEX_LOAD,
STR_BYTE, STR_PTR, ARENA_GET, ARENA_LEN, all arith/bitwise/compare/cast.

`ALLOC_VAR` / `ALLOC_ARRAY` are intentionally listed as side-effecting even
though they have results — line 19-20 comments rationale ("backend uses them
for layout"). Defensible: dropping a dead allocation would shift subsequent
slot indices.

The fixpoint at line 113-125 spreads liveness backward from side-effecting
op operands AND from any op whose result is live. This is correct for a
forward-substituting SSA dataflow.

**No findings.**

---

## Summary

* cycle-86 fix verified clean.
* `mangle_struct` non-injectivity: theoretical collision exists, reachability
  requires adversarial naming + nested generics; confidence ~70%, below
  the 75% audit threshold. Logged as soft observation for cycle-88 or
  deferred-known intake.
* DCE coverage: clean.

**Findings at confidence >= 75%: 0.**

**Verdict: PASS.**

**No edits made to source files.** This audit doc is the only new file.
