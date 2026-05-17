# Stage 36 Post-Increment-10 Audit — Type-Design Lane

**Date**: 2026-05-16
**Auditor**: pr-review-toolkit:type-design-analyzer
**HEAD audited**: `821592f` (Stage 36 Increment 10) + Inc 9 catch-up
ledger `14e1fa4`
**Baseline**: `a451591` (Stage 36 Increment 8)
**Status**: **NOT CLEAN — 1 HIGH + 3 MEDIUM + 2 LOW**

The Inc 9 sweep closed all 8 prior-lane findings (A1 inner-type, A2
arena pair, B1 prove flatten, B2 derive observability, B3 prove src,
B4 to_logic_bool i32, C1 unwrap_logic recovery, C2 derive recovery).
The remaining gaps below are NEW surfaces exposed by Inc 10's
dogfood-driven mixing of handles + chained Logic ops.

## Findings

### A1 HIGH (conf 92) — `register_derivation` handle is a bare `i32`, freely aliasable to any integer

**Files**:
- `helixc/frontend/typecheck.py:2898-2916`
- `helixc/examples/dogfood_09_knowledge_graph.hx:40,44,51-55`

`register_derivation(l, r) -> i32` and the accessors take/return
`TyPrim("i32")` with no nominal wrapper. The Inc 9 A2 fix made
handle 0 the null sentinel via 1-based indexing, but **any i32 can
be passed to `parent_left_at` / `parent_right_at`**: literals, loop
counters, source IDs (themselves i32!). The dogfood `let h_ac: i32 =
register_derivation(1, 2)` then immediately uses `parent_left_at(1)`
would compile cleanly and silently return source-id-of-source-1 —
total provenance fiction. The bounds check is necessary but not
sufficient — it stops out-of-arena reads, not categorically-wrong
reads. Source IDs and derivation handles share the same type and
are routinely adjacent in user code; the confusion is structural.

**Fix family**: introduce `Handle<Derivation>` as a nominal newtype
(`TyHandle(kind="derivation")`) such that `register_derivation`
returns `Handle` and the accessors require it. The runtime
representation stays i32; only the typecheck face changes.
*Architectural — needs user approval.*

### B1 MEDIUM (conf 78) — `if_logic` does not require cond to be `Logic<i32>` and does not enforce then/else inner-type agreement

**File**: `helixc/frontend/typecheck.py:2854-2868`

Per the Inc 9 A1 sweep, `and_logic` / `or_logic` etc. now use
`_is_logic_of(t, "i32")`. `if_logic` was missed: it still uses
`isinstance(t, TyLogic)` for all three args (so cond can be
`Logic<f32>` or `Logic<MyStruct>`), and the return type is
unconditionally `arg_tys[1]` with no check that `then_val.inner ==
else_val.inner`. `if_logic(cond, prove(1, 0), prove(1.0, 0))`
typechecks; downstream code consuming the result as `Logic<i32>`
gets `Logic<f32>` data on the else branch — silent type punning of
the exact shape the Inc 9 A1 fix closed elsewhere.

**Fix family**: tighten cond to `_is_logic_of(t, "i32")`; assert
`arg_tys[1].inner == arg_tys[2].inner` and emit trap 24100 on
mismatch.

### B2 MEDIUM (conf 70) — AD chain rule for boolean Logic ops silently treats them as zero-derivative

**File**: `helixc/frontend/autodiff.py:69-77` (`AD_KNOWN_PURE_CALLS`)

The Inc 9 C2 fix registered `and_logic` / `or_logic` / `not_logic` /
… as AD-pure so let-inlining does not trap. Side effect: a function
`f(x) = and_logic(prove(1, 0), x_as_logic)` differentiated under
`grad_rev` returns 0 with no diagnostic (these ops have no
registered chain rule and pure means "skip"). The author probably
intended `fuzzy_and` (which IS differentiable). The opposite case —
fuzzy op in non-differentiated code — is fine. The asymmetry: boolean
ops in differentiated code should at least warn, ideally refuse.

**Fix family**: distinguish AD-pure-and-zero-derivative from
AD-pure-with-chain-rule. The non-fuzzy Logic ops should raise
`NotImplementedError` like the Inc 9 B3 fix did for `prove(x, x)`,
not silently zero.

### B3 MEDIUM (conf 65) — `derive`'s arena side effect is invisible to typecheck and AD

**Files**: `helixc/ir/lower_ast.py:1836-1870`,
`helixc/frontend/autodiff.py:75`

Inc 9 B2 fix made `derive(a, b)` emit `ARENA_PUSH_PAIR`, so the call
now mutates the global arena. But `derive` is still in
`AD_KNOWN_PURE_CALLS`, and the typecheck signature claims
`Logic<T>`-in-Logic<T>-out with no effect annotation. A user
calling `derive` inside `grad_rev` gets transparent inlining + arena
mutation under-the-hood — debugging arena state in differentiated
code becomes nondeterministic vs. the source. The Inc 9 ledger
acknowledged this as "user-visible behaviour change" but did not
propagate it to AD-purity.

**Fix family**: remove `derive` from `AD_KNOWN_PURE_CALLS` (it now
has the same effect signature as `register_derivation`, which is
*not* in the set either — but `derive` is). Alternatively: split
into `derive_pure` (Phase-1 lattice) and `derive_logged` (current
behavior).

### C1 LOW (conf 60) — `register_derivation` accepts any `_is_int_scalar` for source IDs (i64/u32/u64)

**File**: `helixc/frontend/typecheck.py:2898-2906`

Same family as the Inc 9 B4 fix on `to_logic_bool`: the source-tag
args go through `_is_int_scalar` (i32/i64/u32/u64) but the arena
slots are i32. Passing i64 silently truncates the high 32 bits in
the eventual `ARENA_PUSH_PAIR` store. Symmetric fix to B4.

### C2 LOW (conf 55) — `prove` source-tag also accepts non-i32 ints

**File**: `helixc/frontend/typecheck.py:2708`

`if not self._is_int_scalar(arg_tys[1])` — same i32/i64/u32/u64
acceptance as C1. Phase-0 single-tag provenance is documented as
i32; the AD-side B3 fix already requires the source to be an
IntLit in differentiated code, but the typecheck-side is loose.

## Verified clean

- The Inc 9 A1 sweep correctly tightened all 11 boolean/fuzzy ops
  to `_is_logic_of(t, "i32")` / `_is_logic_of(t, "f32")` —
  symmetric and complete with the single exception of `if_logic`
  (B1 above).
- The Inc 9 B1+C1+C2+B3 recovery / strictness fixes hold: `prove`
  rejects nested Logic, `unwrap_logic` and `derive` return
  `TyUnknown` on type error (no cascading), `prove(x, x)` in
  differentiated code raises with a clear pointer to
  `register_derivation`.
- The Inc 9 A2 1-based-handle invariant: handle 0 reliably returns
  -1 from both accessors via the bounds check (since 0 - 1 = -1
  fails `>= 0`).
- The Inc 10 dogfood (`dogfood_09_knowledge_graph.hx`) compiles and
  runs (exits 42 per its assertion structure) — no typecheck gap
  exposed *for its specific shape*, but A1 is exactly the next gap
  a slightly-evolved dogfood would hit.
- `parent_left_at` / `parent_right_at` bounds check is correctly
  routed through SELECT + CMP_GE/CMP_LE per the Inc 9 A1
  silent-failure fix (out-of-range → -1).

## Verification commands

```
git rev-parse HEAD                        # → 821592f...
git log --oneline a451591..HEAD -- helixc/frontend/typecheck.py
python -m pytest helixc/tests/test_stage36_provenance.py -q
python -m pytest helixc/tests/test_provenance.py -q
python scripts/stage33_selfhost_gate.py
```
