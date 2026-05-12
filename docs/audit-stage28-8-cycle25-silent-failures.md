# Stage 28.8 Pre-29 Audit Gate — Cycle 25 (Audit A: silent failures)

**Date:** 2026-05-11
**HEAD:** `6db467f` ("Audit 28.8 cycle 23+: close C22-C (HIGH, match_lower
walker drift)")
**Lens:** silent failures (Audit A)
**Streak counter at start:** 2/5 (cycle 24 silent-failures was clean).

> Note on HEAD vs. brief: the cycle-25 task brief named the cycle-24
> reference HEAD `89d49e9`. Local repo HEAD is one commit forward at
> `6db467f`, which is an autonomous fix-sweep for cycle-22 Audit C
> finding C22-C (match_lower walker drift). The audit is performed at
> `6db467f` and includes both autonomous commits since the last
> declared-clean baseline (`89d49e9` effect_check.py OP_EFFECTS
> extensions and `6db467f` match_lower.py walker arm extensions) as
> first-class verification targets.

---

## Scope

Strict-criterion read-only audit. Re-scan all surfaces cycle 24 cleared
plus the two new autonomous fix-sweep commits:

1. `helixc/ir/passes/effect_check.py` — `OP_EFFECTS` table additions
   (FFI_CALL, ARENA_PUSH/SET, QUOTE, REFLECT_HASH, TILE_INDEX_STORE,
   TRACE_ENTRY/EXIT) and `callees()` FFI_CALL branch (commit
   `89d49e9`). Verified clean by cycle 24; verify stability holds.
2. `helixc/frontend/match_lower.py` — six new `_rewrite_expr` dispatch
   arms (UnsafeBlock, Range, Modify, Break, Quote, Splice) added in
   commit `6db467f`. Verify the additions do not introduce a new
   silent-fall-through surface.
3. `helixc/tests/test_match.py` — two new C22-C regression tests
   (`test_c22_c_match_inside_unsafe_block_lowered`,
   `test_c22_c_match_inside_range_lowered`). Verify they pin the
   relevant invariant.
4. All cycle-24 Priority 2 targets (ast_walker, x86_64._op_suffix,
   match_lower._FRESH_COUNTER, four walker refactors, isize/usize
   sites in const_fold + x86_64 + ptx, grad_pass) for stability.
5. Wider scan of `helixc/{bootstrap,frontend,ir,backend,stdlib}/`
   for any new silent-failure surface.

No re-flagging of cycle 1-24 findings (per task brief). No
manufacturing of findings.

---

## Priority 1 — Re-scan of `6db467f` match_lower fix-sweep

### Diff inventory

`git diff 89d49e9..6db467f -- helixc/` returns exactly two files:

| File                              | Lines | Kind                  |
|-----------------------------------|-------|-----------------------|
| `helixc/frontend/match_lower.py`  | +34   | Six new walker arms   |
| `helixc/tests/test_match.py`      | +76   | Two regression tests  |

No other production code changed since `89d49e9`.

### Walker arm additions (`match_lower.py:169-202`)

Each new arm is a strict recursion: recurse into every `Expr`-typed
field of the sub-AST node, then return the (possibly-mutated)
expression. Specifically:

- **UnsafeBlock** (line 176-178): recurses into `expr.body` via
  `_rewrite_block`, which itself walks every statement plus
  `final_expr`. No Expr child is skipped.
- **Range** (line 179-184): recurses into both `start` and `end`
  with explicit `None`-guards. The `_rewrite_expr` entry-point at
  line 99-100 also short-circuits on `None`, so the guards are
  defense-in-depth rather than load-bearing — neither path can
  silently swallow a Match.
- **Modify** (line 185-189): recurses into all three fields
  (`target`, `transformation`, `verifier`). The `A.Modify` dataclass
  has no other Expr-typed fields at this stage.
- **Break** (line 193-196): recurses into `value` with explicit
  `None`-guard (Break may carry no value). Same redundancy with the
  entry-point guard as Range.
- **Quote** (line 197-199): recurses into `inner`.
- **Splice** (line 200-202): recurses into `inner`.

The terminal `return expr` at line 203 still serves as the
fall-through for genuinely terminal Expr subtypes (`Name`, `IntLit`,
`FloatLit`, `BoolLit`, `StrLit`, `Path`, etc.) that hold no Expr
children needing rewriting. This is the same fall-through that
cycle-22 Audit C flagged and cycle-23 fix-sweep handled; per the
no-re-flag rule it is a closed cycle-22 finding, not a cycle-25
finding. The cycle-22 forward note about future-AST-author drift
(any new AST subtype with Expr children added after this commit)
remains a forward note, not a cycle-25 finding.

### No new silent surface introduced

Each new arm follows the same pattern as the existing arms
(`Block`, `If`, `Binary`, `Unary`, `Call`, `For`, `While`, `Loop`,
`Cast`, `Assign`, `TupleLit`, `ArrayLit`, `Index`, `Return`,
`StructLit`, `Field`): explicit isinstance check, recurse on every
Expr child, return the node. The dispatch table is closed-form
covered by the six new arms for the AST subtypes inventoried in
the commit message (Stage 28.6 `UnsafeBlock` + earlier `Range` /
`Modify` / `Break` / `Quote` / `Splice`).

I verified by class-grep that no other AST class in `ast_nodes.py`
carries an Expr-typed field beyond what is now dispatched:

- `For.iter_expr` — dispatched at line 132-135.
- `While.cond` — dispatched at line 136-139.
- `If.cond`/`then`/`else_` — dispatched at line 112-120.
- `Call.callee`/`args` — dispatched at line 128-131.
- `Cast.value` — dispatched at line 143-145.
- `Assign.value` — dispatched at line 146-148.
- `Return.value` — dispatched at line 159-161.
- `Field.obj` — dispatched at line 166-168.
- `Range.start`/`end` — newly dispatched at line 179-184.
- `Modify.target`/`transformation`/`verifier` — newly dispatched
  at line 185-189.
- `Break.value` — newly dispatched at line 193-196.
- `Quote.inner` — newly dispatched at line 197-199.
- `Splice.inner` — newly dispatched at line 200-202.
- `UnsafeBlock.body` — newly dispatched at line 176-178 (via
  `_rewrite_block` which walks every stmt + final_expr).

No Expr-bearing subtype now falls through silently. Cycle-22 C22-C
is fully closed.

### `_rewrite_stmt` and `_rewrite_block` unchanged

These call only `_rewrite_expr`, which now covers the prior
silent-drift cases. No new code path here.

### Test coverage

`test_c22_c_match_inside_unsafe_block_lowered` parses
`fn main() -> i32 { unsafe { match 1 { 1 => 42, _ => 0 } } }`,
calls `lower_matches`, and walks the entire post-pass program
asserting no `A.Match` node remains anywhere. This is the exact
inverse of the silent-failure mode (Match persisting past
lower_matches), so the test pins the regression. The walker uses
`vars(node).values()` to recurse, which covers every dataclass
field (no skip-list).

`test_c22_c_match_inside_range_lowered` does the same for
`for i in 0..(match n { 5 => 10, _ => 5 }) { ... }`, exercising
both `Range.end` recursion and the `Range` arm itself.

`pytest helixc/tests/test_match.py helixc/tests/test_effect_check.py
helixc/tests/test_codegen_determinism.py` → 54 passed in 23.58 s
at HEAD `6db467f`.

**Verdict for Priority 1: clean.** The six new arms are pure
extensions of an existing dispatch pattern; they close cycle-22
C22-C without introducing any new silent-failure surface.

---

## Priority 2 — Re-scan of `89d49e9` effect_check fix-sweep

Cycle 24 declared this clean. Re-verify that nothing has shifted at
`6db467f`.

### File unchanged since `89d49e9`

`git diff 89d49e9..6db467f -- helixc/ir/passes/effect_check.py`
returns empty. The seven new OP_EFFECTS entries (FFI_CALL → "ffi";
ARENA_PUSH/SET → "arena"; QUOTE / REFLECT_HASH → "reflect";
TILE_INDEX_STORE → "tile_io"; TRACE_ENTRY/EXIT → "trace") and the
FFI_CALL branch of `callees()` (line 158-166) are byte-identical
to cycle-24 state. Cycle-24 verdict carries: clean.

### `<indirect-ffi>` sentinel path re-verified

`compute_closure` (line 188-201) routes the `<indirect-ffi>`
sentinel through the `else` branch at line 197-199, which adds
`"unknown"` to the closure. This is the same explicit-loud-failure
path as `<indirect>`. Any non-`unknown`-declaring caller fails
loudly at trap 19001. No silent path.

`pytest helixc/tests/test_effect_check.py` → 22 passed (was 18 →
+4 cycle-23 additions, unchanged at cycle 25).

**Verdict for Priority 2: clean.**

---

## Priority 3 — Cycle-24 Priority-2 surface re-scan at HEAD `6db467f`

Each cycle-24 Priority-2 target was re-checked against the
`89d49e9..6db467f` diff. All targets either appear in the diff and
have been re-verified above (match_lower), or are unchanged.

### Target 1 — `helixc/frontend/ast_walker.py`

Unchanged (`git diff 89d49e9..6db467f -- helixc/frontend/ast_walker.py`
returns empty). `_TYPE_FIELD_NAMES` / `_NON_NODE_FIELD_NAMES` skip-
lists, `_is_ast_node` filter, `visit()` / `generic_visit()` dispatch,
and the `if node is None: return None` guard at line 180 are
byte-identical to cycle-24 state. Cycle-24 verdict carries: clean.

### Target 2 — `_op_suffix` and `id(op)` index table

`helixc/backend/x86_64.py:825-876` unchanged. The deterministic
`{fn_index}_{op_index}` form remains the primary path; the
`{fn_index}_unk{id(op):x}` fallback is still intentionally loud
(hex address surfaces in any byte-diff test). Cycle-24 verdict
carries: clean.

### Target 3 — `_FRESH_COUNTER` reset

`helixc/frontend/match_lower.py:52` (`_FRESH_COUNTER[0] = 0` at
every `lower_matches(prog)` entry) appears in the diff context but
the line itself is unchanged. The cycle-23+ commit added arms
*after* line 168; the entry-point reset is untouched. Cycle-24
verdict carries: clean.

### Target 4 — Four walker refactors

`panic_pass.py`, `deprecated_pass.py`, `grad_pass._expr_has_grad`,
`struct_mono.visit_expr` all unchanged (`git diff
89d49e9..6db467f` reports zero changes to any of these). The
`_BodyVisitor` overrides and the byte-identical body-walk pipeline
in `struct_mono.py` carry. Cycle-24 verdict carries: clean.

### Target 5 — isize/usize fixes (cycles 16-21)

`const_fold.py`, `x86_64.py`, `ptx.py` unchanged. All width-keyed
sites enumerated in cycle 24 carry. The cycle-22 forward note about
the `.get(dtype, 4)` / `.get(dtype, "u32")` soft-fallback in
`ptx.py:350-353` (Stage-29-class scalar-width-predicate refactor)
is recorded as a forward note, not a finding. Cycle-24 verdict
carries: clean.

### Target 6 — Deferred grad_pass rewriter + resolver

`grad_pass.py` unchanged. The v0.2 `ASTTransformer` work, plus the
remaining hand-rolled walkers `_resolve_in_expr` and
`_rewrite_in_expr`, are documented forward notes for Stage 28.8.2
continuation, not cycle-25 findings.

---

## Priority 4 — Wider silent-failure scan

Grep across `helixc/` for the canonical silent-failure smells:

- `except\s*:` (bare except) — zero hits in `helixc/`.
- `except\s+Exception\s*:\s*$` (catch-all) — zero hits in production
  code paths. Pre-existing test-helper allowances unchanged.
- `pass\s*#\s*(ignore|skip)` — zero hits.
- `\.get\(.*,\s*0\)` / `\.get\(.*,\s*None\)` as silent type-decay
  → all hits are pre-existing and either covered by explicit
  default-is-correct semantics (META_ATTRS, attribute lookup) or
  already flagged as forward notes in cycles 17-23 (ptx.py).

No new silent-failure smell has been introduced at `6db467f`.

---

## Test verification

At HEAD `6db467f`:

| Suite                          | Tests | Status     |
|--------------------------------|-------|------------|
| `test_match.py`                | 27    | all pass   |
| `test_effect_check.py`         | 22    | all pass   |
| `test_codegen_determinism.py`  | 5     | all pass   |
| Combined (audit-scope)         | 54    | all pass   |

Tests exercising every previously-flagged silent-failure surface
remain green.

---

## Audit findings

**Cycle 25 silent-failures audit: CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

Key observations:

- Two autonomous fix-sweep commits since cycle-24 baseline
  (`89d49e9` effect_check; `6db467f` match_lower walker). Both are
  pure-additive — no existing code path was modified or removed,
  only new dispatch arms / table entries were added.
- The match_lower walker now correctly recurses into every
  Expr-bearing AST subtype currently defined in `ast_nodes.py`.
  Class-grep confirms no remaining drift gap.
- The effect_check `<indirect-ffi>` sentinel routes through the
  same explicit-`unknown` failure path as `<indirect>`. No silent
  surface.
- All previously-clean surfaces (ast_walker, x86_64._op_suffix,
  _FRESH_COUNTER, four walker refactors, isize/usize sites,
  grad_pass) remain unchanged at `6db467f` and carry their
  cycle-24 verdicts.
- No new silent-failure smell exists anywhere in `helixc/` at
  `6db467f`.
- 54 audit-scope tests pass.

**Clean-cycle counter:** was 2/5 → **advances to 3/5.**

Two more consecutive clean cycles required to fire the Stage-29
gate.

---

## Out-of-scope per task instructions

- Effect-label docstring drift in `effect_check.py:15-22` (lists
  four labels, table now carries nine families) — recorded by
  cycle-24 as a forward note, not re-flagged here per the
  no-re-flag rule.
- Stage-29-class "centralize scalar-width predicate" refactor for
  `ptx.py:350-353` — forward note since cycle 17.
- v0.2 `ASTTransformer` base class for `grad_pass._resolve_in_expr`,
  `grad_pass._rewrite_in_expr`, and `match_lower._rewrite_expr` —
  documented future migration target (Stage 28.8.2 continuation),
  not a finding.
- Future-AST-author drift risk on any hand-rolled walker — forward
  note from cycle 22, not a cycle-25 finding.

---

## Files touched by this audit

None — read-only. Only this doc.

## Cross-reference

- Cycle 24 silent-failures (declared CLEAN, advanced 1/5 → 2/5):
  `docs/audit-stage28-8-cycle24-silent-failures.md`.
- Cycle 23 silent-failures (declared CLEAN):
  `docs/audit-stage28-8-cycle23-silent-failures.md`.
- Cycle 24 HEAD: `89d49e9`; cycle 25 HEAD: `6db467f` (cycle 23+
  match_lower walker fix-sweep commit).
- Production-code delta scope vs. cycle-24 baseline:
  `helixc/frontend/match_lower.py` (+34) and
  `helixc/tests/test_match.py` (+76) only.
- Test suite verification: 54 audit-scope tests pass.
