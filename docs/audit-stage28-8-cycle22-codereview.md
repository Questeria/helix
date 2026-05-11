# Stage 28.8 Cycle 22 — Code-Review Audit (Audit C)

**Date**: 2026-05-11
**Audit HEAD**: `bee36e6` — "Audit 28.8 cycle 21 fix-sweep: close
C20-1 (HIGH, PTX backend isize/usize silent 32-bit)". Confirmed
no production-code change since cycle 21 via
`git log bee36e6..HEAD --oneline` (empty) and
`git diff bee36e6 HEAD --stat` (empty).
**Reviewer**: code-review lens (third audit pass of cycle 22).
**Strict criterion**: cycle counts CLEAN only when **zero findings ≥
80% confidence** of ANY severity.
**Streak counter at start**: 1/5 (cycle 21 was CLEAN; cycle-22
Audit A — silent failures — declared CLEAN and advanced to 2/5;
this Audit C is the third and final pass of cycle 22).

---

## Scope

Per the cycle-22 brief: **fresh-eyes re-audit** with a new class of
adversarial probe (generic struct + grad rewrite, TraceBuffer +
struct, @autotune + match, @kernel + pytree). Read-only.

Prior cycle-21 code-review audit pinned only the cycle-21 fix-sweep
(four-site PTX width-table extension for isize/usize). With no
production-code change since, the fix-sweep stays verified-clean.
The cycle-22 lens shifts: re-examine the cross-pass interactions
between recently-refactored frontend passes (Stage 28.8.2 shared
ASTVisitor library + downstream walker migrations) and the
**older, hand-rolled** walkers that did **not** get migrated —
because those are precisely where the cycle-2 C2-4 / cycle-3 C3-1
defect class (incomplete `_rewrite_in_expr` arm coverage) has
historically lived.

The cycle-22 Audit A (silent-failures) audit already confirmed
that the 4 walkers that **did** migrate to ASTVisitor (panic,
deprecated, struct_mono.visit_expr, grad_pass._expr_has_grad) are
all clean. The remaining bespoke walkers in the frontend are:

- `grad_pass._rewrite_in_expr` (cycle 2 C2-4 fixed; cycle 22 Audit A
  recorded "exhaustive against `ast_nodes.Expr` subtypes")
- `grad_pass._resolve_in_expr` (same; cycle 2 C2-4 fixed)
- `match_lower._rewrite_expr` (**not previously audited end-to-end
  against the full ``ast_nodes.Expr`` subtype set**)

Fresh-eyes probe targets:

1. Generic struct + grad rewrite end-to-end pipeline
2. TraceBuffer + struct-type field behavior
3. @autotune dedup + variant_count interaction with cap diag
4. @kernel + pytree cycle / depth guard
5. **`match_lower._rewrite_expr` AST-subtype completeness**

---

## Method

1. **Confirm no production-code change since cycle 21**:
   - `git log bee36e6..HEAD --oneline` → empty.
   - `git diff bee36e6 HEAD --stat` → empty.
   - `git status` → 4 untracked audit-doc files (this doc + the
     three cycle-22/cycle-19/cycle-21 docs); no `M` lines for
     `helixc/`.
2. **Read** the prior cycle-22 Audit A doc (silent failures) to
   avoid duplicating findings. Audit A returned CLEAN; this doc
   takes the code-review lens.
3. **Enumerate Expr subtypes** in `helixc/frontend/ast_nodes.py`:
   `ArrayLit, Assign, Binary, Block, BoolLit, Break, Call, Cast,
   CharLit, Continue, Field, FloatLit, For, If, Index, IntLit,
   Loop, Match, Modify, Name, Path, Quote, Range, Return, Splice,
   StrLit, StructLit, TileLit, TupleLit, Unary, UnsafeBlock,
   While` (32 total).
4. **Adversarial cross-pass probes** under the four prompt-suggested
   classes (generic struct + grad, TraceBuffer + struct, @autotune
   + match, @kernel + pytree). Plus a fifth probe born from
   reading `match_lower._rewrite_expr` against the Expr-subtype
   enumeration: build short repro programs with `match` placed
   inside every Expr-subtype slot that ``_rewrite_expr`` does not
   recurse into.
5. **End-to-end pipeline check**: for each suspected miss, drive
   `python -m helixc.backend.x86_64 <input.hx> <output.bin>` and
   observe whether the program reaches IR-lower's
   `assert ... A.Match should not reach _lower_expr` guard.

---

## Adversarial probes — pass-by-pass summary

### Probe 1 — Generic struct + grad rewrite end-to-end

Program:

```
struct Pair[T] { x: T, y: T }
fn loss(x: f32) -> f32 { x * x }
fn use_grad(p: Pair<f32>) -> f32 {
  let g = grad(loss);
  g(p.x)
}
```

Pipeline: `parse → grad_pass → monomorphize_structs` produces:

- `loss__grad` synthesized
- `Pair__f32` mono'd
- typecheck clean

This composition is **correct**. The let-alias rewrite + struct
monomorphization commute as expected.

**Observation (not a finding)**: `_generate_grad_fn` at
`grad_pass.py:629-634` rewrites every parameter type to
`TyName("f32")` unconditionally, even when the original parameter
type is a struct (e.g. `Pair<f32>`). This is documented as
"gradient takes plain floats" but produces an arity-mismatched
signature relative to the original fn: a user writing
`grad(use_struct)` where `use_struct(p: Pair<f32>) -> f32` gets
back `use_struct__grad(p: f32)`. This is **intended Phase-0
behavior** (see the comment at line 629), is documented, and the
call-site responsibility is on the user; **not a code-review
finding**.

### Probe 2 — TraceBuffer push + struct operands

`TraceBuffer.push` correctly traps at cap; `TraceEvent` is
`frozen=True` and stores operands as a tuple — but the tuple may
itself contain non-hashable values (lists, dicts). Since
`TraceEvent` is never used as a dict key in the current codebase
and `__eq__` on `frozen=True` dataclasses uses structural
equality (not hashing), this is **not a defect** — just a
documented limitation. **Not a finding**.

### Probe 3 — @autotune dedup + variant_count cap interaction

`variant_count({'A': [...], 'B': [...], 'C': []}) == 0` —
including an empty per-key list collapses the product to zero,
which would in principle bypass the `MAX_VARIANT_PRODUCT > 16`
diagnostic. However, `validate_autotune` at lines 213-218 emits
a dedicated "parameter 'C' has empty value list" diag *before*
the cap check, so the user always sees the empty-list complaint.
**Not a finding**.

`autotune_variants({'B':[16,32], 'A':[4,8]})` and
`autotune_variants({'A':[4,8], 'B':[16,32]})` produce different
key insertion order in the variant dicts. But: `mangled_variant_name`
at `autotune.py:175` sorts keys before joining, so the produced
variant **names** are insertion-order-independent. The variant
**dict** ordering does depend on insertion order, but iteration
order at codegen is consumed by name (mangled) and the iteration
itself is parser-deterministic (parser builds `fn.attrs` in
source order). **Not a finding** for determinism in single-process
codegen; flagged in Audit A as a future-codegen-concurrency
concern but already noted out-of-scope.

### Probe 4 — @kernel + pytree depth + cycle guard

`flatten_pytree` on a self-referencing struct (`struct Self {
val: f32, child: Self }`) raises trap 26003 cleanly. Depth-cap
fires at trap 26001 for straight-line depth-5+ chains
(`_unflatten` also has the cycle-2 deferred-17 depth guard at
line 246, symmetric with flatten). D-wrapped vs bare-float
distinction handled correctly by the cycle-3 B9(3) split. **Not
a finding**.

### Probe 5 — `match_lower._rewrite_expr` AST-subtype completeness

`match_lower._rewrite_expr` at `helixc/frontend/match_lower.py:96-169`
uses a hand-rolled `isinstance` cascade over Expr subtypes —
**not** migrated to `ASTVisitor` in the Stage 28.8.2 sweep. The
module docstring (line 7) claims: *"After this pass, no
`Match`/`MatchArm`/`PatRange`/`PatOr` nodes remain in the tree,
so the rest of the pipeline (typecheck second pass, autodiff, IR
lowering) can remain match-agnostic."*

That claim is contract-load-bearing: IR-lower at
`helixc/ir/lower_ast.py` carries an
`assert ... A.Match should not reach _lower_expr — match_lower
must rewrite it to if/let chains first` guard. So any node type
that holds an `Expr` child but is not recursed into by
`_rewrite_expr` is a **silent contract violation** at
match_lower's exit, surfaced as an `AssertionError` at IR-lower.

Cross-check `_rewrite_expr` arms (lines 96-169) against the
ast_nodes.Expr enumeration:

| Expr subtype  | Has Expr child? | Handled by `_rewrite_expr`? |
|---------------|------------------|------------------------------|
| Match         | yes (recursive) | yes (line 101)               |
| Block         | yes             | yes (line 109)               |
| If            | yes             | yes (line 112)               |
| Binary        | yes             | yes (line 121)               |
| Unary         | yes             | yes (line 125)               |
| Call          | yes             | yes (line 128)               |
| For           | yes             | yes (line 132)               |
| While         | yes             | yes (line 136)               |
| Loop          | yes             | yes (line 140)               |
| Cast          | yes             | yes (line 143)               |
| Assign        | yes             | yes (line 146)               |
| TupleLit      | yes             | yes (line 149)               |
| ArrayLit      | yes             | yes (line 152)               |
| Index         | yes             | yes (line 155)               |
| Return        | yes (Optional)  | yes (line 159)               |
| StructLit     | yes             | yes (line 162)               |
| Field         | yes             | yes (line 166)               |
| **UnsafeBlock** | **yes (body: Block)** | **NO** — falls through |
| **Range**     | **yes (start, end: Optional[Expr])** | **NO** |
| **Modify**    | **yes (target, transformation, verifier)** | **NO** |
| Break         | yes (Optional)  | NO — but see below           |
| Quote         | yes (inner: Expr)| NO — but see below          |
| Splice        | yes (inner: Expr)| NO — but see below          |
| TileLit       | yes (shape, memspace) | NO — but see below      |
| BoolLit/IntLit/FloatLit/CharLit/StrLit/Name/Continue/Path | no | trivially clean |

**Reachable end-to-end crashes confirmed**:

Three of the un-recursed-into arms produce a reachable
internal-compiler crash via the production `helixc.backend.x86_64`
CLI driver.

#### Repro A — `match` inside `unsafe { ... }`

```
fn f(x: i32) -> i32 {
  unsafe {
    match x { 0 => 100, _ => 200, }
  }
}
fn main() -> i32 { f(0) }
```

```
$ python -m helixc.backend.x86_64 repro_a.hx repro_a.bin
...
AssertionError: A.Match should not reach _lower_expr — match_lower
must rewrite it to if/let chains first. Got at 4:5. If you've
added a new AST item type that holds expressions, extend
lower_matches.
[returncode: 1]
```

The IR-lower error message **itself** documents the defect:
"If you've added a new AST item type that holds expressions,
extend lower_matches." Stage 28.6 added `UnsafeBlock` to
`ast_nodes.py` (line 205) but `match_lower._rewrite_expr` was
not extended to recurse into `UnsafeBlock.body`.

#### Repro B — `match` inside `Range.end`

```
fn f(n: i32) -> i32 {
  let mut total = 0;
  for i in 0..match n { 0 => 10, _ => 20 } {
    total = total + i;
  };
  total
}
fn main() -> i32 { f(0) }
```

Same `AssertionError` at IR-lower. `For.iter_expr` is recursed
into by `_rewrite_expr` (line 133), but when that iter_expr is a
`Range(start, end)`, `_rewrite_expr` falls through the bottom
`return expr` at line 169 without recursing into `Range.start`
or `Range.end`.

#### Repro C — `match` inside `Modify(...)`

```
fn always_ok(x: i32) -> bool { true }
fn step(x: i32) -> i32 { x + 1 }
fn f(x: i32) -> i32 {
  modify(x, match x { 0 => step, _ => step }, always_ok)
}
fn main() -> i32 { f(0) }
```

Same `AssertionError`. `Modify.{target, transformation, verifier}`
are all Expr-typed and not recursed into.

**Status of the remaining missed arms**:

- `Break.value`: parseable as `break match x { ... };` but
  reaching IR-lower path requires the break-value to actually be
  evaluated. Empirical test: `loop { break match x { ... } }`
  reaches `lower OK` (no crash), so IR-lower's break path does
  not currently dispatch through `_lower_expr` for break values.
  **Latent** — if a future change wires `break.value` through
  `_lower_expr`, the same defect class would surface.
- `Quote.inner`: a `match` inside a `quote { ... }` body reaches
  `lower OK` because Quote bodies are not lowered as live
  expressions in Phase-0. **Latent** — same class as Break.
- `Splice.inner`: not currently reachable through `_lower_expr`
  in a way that triggers Match-dispatch (splice replays an
  AstNode value). **Latent**.
- `TileLit.shape` / `TileLit.memspace`: phase-0 shape parsing
  rejects non-const expressions; `match` in a tile shape is not
  parseable. **Not reachable**.

**Confidence**: **≥ 95%**. Three independent reachable
end-to-end crashes confirmed via the production codegen CLI;
all three produce the identical `AssertionError` at
`lower_ast.py` whose message explicitly diagnoses the defect.

**Defect class lineage**: this is the **same shape** as cycle-2
finding C2-4 (`grad_pass._rewrite_in_expr` / `_resolve_in_expr`
missing arms) — which was scored **HIGH**. Cycle-2 C2-4's fix
extended grad_pass to cover Match / Loop / Field / Return /
Break / Assign / Range / StructLit / TupleLit / ArrayLit /
UnsafeBlock / Quote / Splice / Modify (per the comment block at
`grad_pass.py:164-170`). The cycle-2 sweep did **not** apply the
same lesson to `match_lower._rewrite_expr`, which has
structurally the same hand-rolled cascade and the same drift
exposure. The Audit-A doc for cycle 22 noted that the deferred
`_rewrite_in_expr` and `_resolve_in_expr` are tracked as v0.2
ASTTransformer work — but that observation was scoped to
*grad_pass*, not to *match_lower*, even though match_lower is
the **earlier** pass (runs before grad_pass for the typecheck-
seen Match form).

**Severity**: **HIGH** (same as the cycle-2 C2-4 analog). The
reasoning:

- It is **reachable from valid user source** through three
  different language features (`unsafe`, `Range`, `Modify`).
- It produces an **internal compiler error** (AssertionError),
  not a user-friendly diagnostic — the user sees a Python
  traceback from the codegen CLI.
- It is a **silent contract violation** at match_lower's exit
  boundary (the module docstring promises no Match nodes
  remain).
- It is **drift-prone**: every new Expr subtype added to
  `ast_nodes.py` re-opens the window unless the author also
  edits `match_lower._rewrite_expr`. UnsafeBlock (Stage 28.6),
  Range (earlier), and Modify (AGI-specific) all post-date the
  original match_lower walker design.

---

## Finding C22-1 / HIGH — `match_lower._rewrite_expr` missing arms for UnsafeBlock / Range / Modify silently leave Match nodes in the AST, crashing IR-lower

**Location**:
- `helixc/frontend/match_lower.py:96-169` (`_rewrite_expr`)
- Reachable crash site: `helixc/ir/lower_ast.py` (the
  `assert ... A.Match should not reach _lower_expr` guard)

**Severity**: HIGH (precedent: cycle-2 C2-4 in `grad_pass`)
**Confidence**: 95%
**Category**: silent miss / contract violation / drift-prone walker

**Description**:

The match-lowering pass walks every Expr node in a fn body to
desugar `Match` into nested `if/let` chains. Its dispatch in
`_rewrite_expr` (lines 96-169) is a hand-rolled `isinstance`
cascade over Expr subtypes. The dispatch **does not handle**
three Expr subtypes that hold Expr children:

- `A.UnsafeBlock` (body: Block) — added Stage 28.6
- `A.Range` (start, end: Optional[Expr])
- `A.Modify` (target, transformation, verifier: Expr) — AGI

When a `Match` expression appears inside any of those positions,
`_rewrite_expr` falls through to the bottom `return expr` at line
169 **without recursing into the child**. The Match node persists
into the post-`lower_matches` AST. The downstream IR-lower's
guard asserts and crashes with an internal-error Python
traceback — not a user-friendly diagnostic.

The module docstring (line 7) explicitly promises this cannot
happen: *"After this pass, no `Match`/... nodes remain in the
tree."* The promise is violated.

**Reproducer A (UnsafeBlock)**:

```
fn f(x: i32) -> i32 {
  unsafe {
    match x { 0 => 100, _ => 200, }
  }
}
fn main() -> i32 { f(0) }
```

```
$ python -m helixc.backend.x86_64 repro.hx repro.bin
AssertionError: A.Match should not reach _lower_expr — match_lower
must rewrite it to if/let chains first. Got at 4:5.
[returncode: 1]
```

**Reproducer B (Range.end)**:

```
fn f(n: i32) -> i32 {
  let mut total = 0;
  for i in 0..match n { 0 => 10, _ => 20 } {
    total = total + i;
  };
  total
}
fn main() -> i32 { f(0) }
```

Same AssertionError; `Got at 4:15`.

**Reproducer C (Modify.transformation)**:

```
fn always_ok(x: i32) -> bool { true }
fn step(x: i32) -> i32 { x + 1 }
fn f(x: i32) -> i32 {
  modify(x, match x { 0 => step, _ => step }, always_ok)
}
fn main() -> i32 { f(0) }
```

Same AssertionError; `Got at 6:13`.

**Hidden errors**:

- User-facing: an internal Python traceback instead of a Helix
  diagnostic when a perfectly valid `unsafe { match x { ... } }`
  program is compiled.
- Fragility: every new Expr subtype added to `ast_nodes.py`
  re-introduces the same window unless the author remembers to
  edit `match_lower._rewrite_expr`. This is exactly the drift
  class that Stage 28.8.2 ASTVisitor migration was built to
  eliminate — `match_lower._rewrite_expr` was simply not on the
  Stage 28.8.2 migration list.
- Latent: Break.value / Quote.inner / Splice.inner currently
  reach `lower OK` because IR-lower's *own* dispatch does not
  walk those child slots, but if a future change wires any of
  them through `_lower_expr`, the same defect class fires
  immediately.

**Recommended fix** (Stage 28.8.2-style):

Either (a) **add three arms** to `_rewrite_expr` matching the
pattern at lines 109-167 (recurse `UnsafeBlock.body` via
`_rewrite_block`; recurse `Range.start`/`Range.end` if
non-None; recurse the three Expr slots of `Modify`); or (b)
**migrate `_rewrite_expr` to `ASTTransformer`** when that base
class lands (the v0.2 work flagged in cycle 22 Audit A for the
grad_pass deferred walkers). Option (a) is the minimal-diff fix
that closes the three reachable crashes now; option (b) is the
durable drift-proof solution.

**Out-of-scope for this read-only audit**: implementing the fix.
Recorded here for the cycle-22 fix-sweep.

---

## Findings

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 1     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **1** |

| ID     | Severity | Location | Defect | Confidence |
|--------|----------|----------|--------|------------|
| C22-1  | HIGH     | `match_lower.py:96-169` | UnsafeBlock / Range / Modify missing from `_rewrite_expr` dispatch — reachable IR-lower AssertionError | 95% |

---

## Cross-lens corroboration

Cycle 22's three lenses (Audit A silent-failures, Audit B
type-design, Audit C code-review) target the same HEAD at
`bee36e6` from three independent methodologies. Audit A
(cycle-22 silent-failures) declared CLEAN and advanced the
counter from 1/5 to 2/5. This Audit C surfaces **C22-1 / HIGH**.

By the strict criterion (zero findings ≥ 80% confidence), cycle
22 is **NOT CLEAN**.

The cycle-22 Audit A did examine the walker-refactor space —
its Target 4 covered the **4 walkers that migrated to
ASTVisitor**. C22-1 is a fresh-eyes finding in a **non-migrated**
hand-rolled walker that was outside cycle-22 Audit A's named
scope (Audit A's Target 6 covered grad_pass's deferred bespoke
walkers — `_rewrite_in_expr` + `_resolve_in_expr` — but not
`match_lower._rewrite_expr`, which has the same bespoke shape
and the same drift exposure).

This is not double-counting: Audit A correctly observed that
grad_pass's two bespoke walkers were exhaustive at HEAD per
cycle-2 C2-4's fix. The same observation **does not transfer**
to match_lower: nobody has previously enumerated match_lower's
walker arms against the full `ast_nodes.Expr` subtype set, and
match_lower was not on the cycle-2 fix-sweep.

---

## Verdict

**Cycle 22 code-review audit: NOT CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 1     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **1** |

C22-1 (HIGH, 95% confidence) — `match_lower._rewrite_expr`
missing arms for UnsafeBlock / Range / Modify cause reachable
internal-compiler AssertionError at IR-lower from valid user
source.

**Clean-cycle counter**: was 2/5 (after Audit A) → **resets to
0/5** per the strict criterion. The cycle-22 fix-sweep should
close C22-1, then a new clean cycle 23 starts the streak fresh.

---

## Files touched by this audit

None — read-only audit. Reproducers above are inline.

---

## Cross-reference

- Cycle 21 code-review (last clean cycle):
  `docs/audit-stage28-8-cycle21-codereview.md`
- Cycle 22 silent-failures (CLEAN, advanced 1/5 → 2/5):
  `docs/audit-stage28-8-cycle22-silent-failures.md`
- Cycle-2 silent-failures C2-4 (precedent for HIGH severity on
  the same defect class):
  `docs/audit-stage28-8-cycle2-silent-failures.md` (Finding C2-4,
  line 280+)
- Cycle-2 C2-4 fix (grad_pass `_rewrite_in_expr` + `_resolve_in_expr`
  extended): see `helixc/frontend/grad_pass.py:164-170` audit
  stamp.
- AST schema (Expr subtypes enumerated): `helixc/frontend/ast_nodes.py`
  (UnsafeBlock at line 205; Range at line 322; Modify at line 377)
- Match-lower contract claim:
  `helixc/frontend/match_lower.py:1-29` (module docstring)
- IR-lower defensive guard (the crash site):
  `helixc/ir/lower_ast.py` (`A.Match should not reach _lower_expr`)
- Pre-existing bootstrap-kovc failure (orthogonal to C22-1, not
  re-flagged): `test_bootstrap_kovc_full_pipeline_arithmetic` at
  HEAD `bee36e6`.
