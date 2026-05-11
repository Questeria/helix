# Stage 28.8 Pre-29 Audit Gate — Cycle 6, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: c3f26ef (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-6's fix-sweep
(c3f26ef — C4-2 REVERT + C5-1..C5-4 + F1..F6 + F2). The cycle-6
fix-sweep touches:

- `helixc/frontend/typecheck.py:2169-2179` — new TyVar/TySize
  cascade-safe arm at the top of `_compatible` (closes cycle-5 F1).
- `helixc/frontend/typecheck.py:1349-1371` — D-domain binop gate
  broadened from `inner_mismatch`-only to
  `inner_mismatch OR (l_is_diff != r_is_diff)` (closes cycle-5 F2;
  symmetric with cycle-4 E2's Logic broadening).
- `helixc/frontend/autodiff.py:709-721` — `_inline_lets` TileLit
  arm changed from identity to walk arm over `shape` + `memspace`
  (closes cycle-5 F4).
- `helixc/backend/x86_64.py:3022-3036` — driver now aborts
  (`sys.exit(1)`) on any `mono_diags` rather than printing
  `warning:` and continuing (closes cycle-5 F3 severity gap).
- `helixc/frontend/typecheck.py:591,596,606,611,2109,2122` — trap
  ids `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` / `TRAP_CAST_MATRIX_
  RECURSION_DEPTH` switched from literal "28802"/"28803" to
  f-string interpolation of constants (audit-C C5-1 hygiene).
- `helixc/frontend/monomorphize.py:117,125` — same for
  `TRAP_SHAPE_FOLD_ZERO_DIV` (28801).
- `helixc/bootstrap/parser.hx:2330-2349` — REVERT of cycle-4 C4-2
  broadening; back to the cycle-2 D2 narrow Call-only sentinel
  (closes cycle-5 F5 and F6, both of which were escalations of
  the C4-2 broadening's contract surface).
- `helixc/tests/test_typecheck.py` + `helixc/tests/test_autodiff.py`
  — regression tests for C4-1, C4-3, C5-2, C4-4, C6-revert.

**Method**: walked each touched code site through the cycle-5
findings list to verify the fix actually closes its stated
contract; then walked each fix forward to look for new contract
surface it might have opened. Mentally executed the F2 D-gate
broadening on the symmetric matrix of (l, r) ∈ {bareT, D<T>,
Logic<T>, D<Logic<T>>} × same, to find diagnostic-text or
sequencing gaps. Re-verified the F1 cascade-safe arm's
interaction with TyArray.size compare and with TyMemTier's
intentional strict separation. Walked TileLit through
`_inline_lets` with both Name-typed shape elements and Name-typed
memspace. Verified the driver-abort path on a ShapeFoldError
reproducer. Cross-checked that the C4-2 REVERT actually removes
the cycle-5 F5/F6 surface (it does — those findings were both
about the C4-2-broadening branches that no longer exist).

**Result**: **2 new findings (0 HIGH, 0 MEDIUM, 2 LOW)**. Cycle 6
addressed every cycle-5 finding cleanly at the contract level —
F1's cascade-safe arm makes the cycle-4 E1 reproducer actually
pass; F2's D-gate broadening mirrors E2's Logic broadening
exactly; F3's driver-abort closes the silent-continue path; F4's
walk arm matches the cycle-5 doc's recommended fix; F5+F6 are
both eliminated by the C4-2 REVERT (the broken broadening branches
that escalated them are gone). However, the F2 D-gate broadening
introduced a small diagnostic-text accuracy gap on mixed-domain
operands (`D<T> + Logic<T>` and `Logic<T> + D<T>` now warn with
`(one side D-wrapped, other bare)` despite the "bare" side being
Logic-wrapped, not bare). The F1 cascade-safe arm is correct for
TyVar/TySize-vs-anything BUT it interacts with the TyMemTier arm
later in `_compatible` in a way that masks a kind-mismatch case
(`_compatible(TyVar('T'), TyMemTier(Lt, ...))` returns True even
though the TyMemTier arm just below would have returned False;
in practice this is correct cascade-safe behavior for generic
substitution, but it weakens the TyMemTier strict-separation
contract slightly).

Zero of the two new findings are stop-the-line; both are LOW
diagnostic-quality items where a fired warning has slightly
imprecise text. The strict criterion ("zero findings of any
severity") is NOT met. **Cycle 6 status**: 2 findings (0 HIGH,
0 MEDIUM, 2 LOW) means cycle 6 does **NOT** count clean. See
"Cycle 6 status" final section.

---

## Summary table

| ID  | Severity | Component                                      | Issue (short)                                                                |
|-----|----------|------------------------------------------------|------------------------------------------------------------------------------|
| G1  | LOW      | typecheck binop D-gate (F2) diagnostic text    | `D<T> + Logic<T>` (or reverse) now warns from the D-arm with `"(one side D-wrapped, other bare)"` text — the "bare" side is actually Logic-wrapped. Pre-cycle-6 silent; cycle-6 warns but with imprecise text. |
| G2  | LOW      | typecheck `_compatible` TyVar/TySize cascade   | Cascade arm at line 2178-2179 runs BEFORE TyMemTier arm at 2182-2185, so `_compatible(TyVar('T'), TyMemTier(...))` returns True, masking the TyMemTier "no cross-tier mixing" contract for the generic-substitution case |

---

## Per-finding sections

### Finding G1: F2 D-gate broadening fires correctly on `D<T> + Logic<T>` but warning text says "bare" when the other side is Logic-wrapped

**File**: `helixc/frontend/typecheck.py:1349-1371` (D-gate broadened in F2);
`helixc/frontend/typecheck.py:1372-1399` (Logic gate from E2 — same
pattern).
**Severity**: LOW
**Category**: diagnostic-text imprecision in a newly-broadened gate

**Description**:
Cycle-6 F2 broadened the D-domain gate from `inner_mismatch`-only
to `inner_mismatch OR (l_is_diff != r_is_diff)`, mirroring the
cycle-4 E2 broadening for Logic. This correctly closes the
same-inner-asymmetric case `D<f64> + f64` (which was silent
pre-fix). However, the broadened gate now also fires on
`D<f64> + Logic<f64>` (and the reverse) because `l_is_diff !=
r_is_diff` is True for that pair too. The D-arm's extra text is:

```python
extra = ""
if not (l_is_diff and r_is_diff):
    extra = " (one side D-wrapped, other bare)"
```

For `D<f64> + Logic<f64>`, `l_is_diff=True`, `r_is_diff=False`,
so `not (True and False) = True` → extra = "(one side D-wrapped,
other bare)". But the *other* side is `Logic<f64>`, which is
not bare. The user sees a warning text that misstates the
operand's domain.

The Logic-arm at 1372-1399 has the same potential issue but is
never reached for this case because the D-arm catches it first
(elif chain ordering at line 1349 → 1372). The Logic-arm is
only reached when BOTH sides have no D wrapper.

Trace `D<f64> + Logic<f64>`:
- l = TyDiff(TyPrim(f64)). l_is_diff=True. l_is_logic = False
  (TyDiff with non-Logic inner).
- r = TyLogic(TyPrim(f64)). r_is_diff=False. r_is_logic=True.
- l_inner = _unwrap(l) = TyPrim(f64). r_inner = TyPrim(f64).
- inner_mismatch = False (same TyPrim).
- D-gate: `(True or False) AND (False or (True != False))` →
  `True AND True` = True. ENTERS D-arm.
- extra = " (one side D-wrapped, other bare)" — INACCURATE.
- _ad_warn_mixed_inner emits the warning.

Pre-cycle-6, this case was silent (D-gate required
inner_mismatch, which was False here). Post-cycle-6, the case
warns with imprecise text. Net is still an improvement (no
longer silent), but the text accuracy regressed.

The full domain matrix on same-inner pairs is:

| l           | r           | l_is_diff | r_is_diff | l_is_logic | r_is_logic | Gate-D fires | Gate-Logic fires | Extra text accurate? |
|-------------|-------------|-----------|-----------|------------|------------|--------------|------------------|----------------------|
| bareT       | bareT       | F         | F         | F          | F          | no           | no               | n/a (no fire)        |
| D<T>        | D<T>        | T         | T         | F          | F          | no (`!=` is F, inner_mismatch F) | n/a (elif) | n/a |
| Logic<T>    | Logic<T>    | F         | F         | T          | T          | no           | no               | n/a |
| D<T>        | bareT       | T         | F         | F          | F          | YES (`!=`)   | n/a              | YES ("bare") |
| Logic<T>    | bareT       | F         | F         | T          | F          | no           | YES (`!=`)       | YES ("bare") |
| D<T>        | Logic<T>    | T         | F         | F          | T          | YES (`!=`)   | n/a              | **NO** ("bare" but actually Logic) |
| Logic<T>    | D<T>        | F         | T         | T          | F          | YES (`!=`)   | n/a              | **NO** ("bare" but actually D) |
| D<Logic<T>> | bareT       | T         | F         | T          | F          | YES (`!=`)   | n/a              | YES ("bare") — but l is double-wrapped, not just D; text says "D-wrapped" which is half-right |
| D<Logic<T>> | Logic<T>    | T         | F         | T          | T          | YES (`!=`)   | n/a              | text says "D-wrapped, other bare" — r is Logic-wrapped, not bare. Same issue. |

Two of the eight rows (D+Logic and Logic+D) have inaccurate
text. A third row (D<Logic> + bareT) has half-accurate text
("D-wrapped" but really "D over Logic").

**Reproducer**:
```python
from helixc.frontend.typecheck import (
    TypeChecker, TyDiff, TyLogic, TyPrim,
)
from helixc.frontend import ast_nodes as A, autodiff as ad
ad._DIFF_WARNINGS = []
tc = TypeChecker(A.Program(span=A.Span(0,0), items=[]))
# Construct a synthetic binop AST with l: D<f64>, r: Logic<f64>
# and call tc._check_expr through the binop arm; observe that
# ad._DIFF_WARNINGS contains a warning string with text:
#   "AD: D-binop with mixed inner types D<f64> vs Logic<f64> —
#    widened to D<f64> (trap 24200/AD002) (one side D-wrapped,
#    other bare)"
# expected text: "(one side D-wrapped, other Logic-wrapped)"
# or some variant naming the actual wrap kind.
```

Source-level:
```helix
fn f() -> D<Logic<f64>> {
    let a: D<f64> = ...;
    let b: Logic<f64> = ...;
    a + b   // warns; result D<Logic<f64>>; warning text says
            // "(one side D-wrapped, other bare)" — inaccurate
}
```

**Recommended fix**:
Refine the extra-text logic in the D-arm to distinguish
"D + bareT" from "D + Logic". One option:

```python
if not (l_is_diff and r_is_diff):
    if l_is_logic or r_is_logic:
        # The non-D side is Logic-wrapped, not bare.
        extra = " (one side D-wrapped, other Logic-wrapped)"
    else:
        extra = " (one side D-wrapped, other bare)"
```

Apply the symmetric refinement to the Logic-arm's extra at line
1394-1395 for completeness, though the Logic-arm only fires when
both sides lack D, so the only same-inner Logic-arm case is
`Logic<T> + bareT` — extra text "(one side Logic-wrapped, other
bare)" is already accurate.

Alternative (less invasive): drop the `" (one side X-wrapped,
other bare)"` extra entirely and let the main warning text
(`mixed inner types {l} vs {r}`) carry the operand-domain
information. The formatted `l` and `r` already include the
TyDiff/TyLogic wrappers (per `_fmt`), so the user can see the
domain mismatch in the main text without an extra-text
disambiguation. This trades a small loss of named-case
emphasis for a guarantee of accuracy.

Severity rationale: the warning still fires (correct contract
strength). Only the supplementary text is off. No silent miscompile;
the user-facing message is just slightly misleading. LOW.

---

### Finding G2: `_compatible` TyVar/TySize cascade arm bypasses the TyMemTier strict-separation arm — `_compatible(TyVar('T'), TyMemTier(Lt, ...))` returns True

**File**: `helixc/frontend/typecheck.py:2166-2185` (cascade-safe arm at
2178-2179 placed before TyMemTier arm at 2182-2185).
**Severity**: LOW
**Category**: ordering interaction — F1's cascade overrides a more-
strict downstream arm for the generic-substitution case

**Description**:
Cycle-6 F1 added a TyVar/TySize cascade-safe arm at the top of
`_compatible`:

```python
def _compatible(self, a: Type, b: Type) -> bool:
    if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
        return True
    # cycle 6 F1
    if isinstance(a, (TyVar, TySize)) or isinstance(b, (TyVar, TySize)):
        return True
    # ... TyMemTier arm ...
    if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
        return a.tier == b.tier and self._compatible(a.inner, b.inner)
    if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
        return False
```

The TyMemTier arm explicitly says "Memory-tier types are
incompatible across tiers (must explicitly consolidate / recall
to convert)". The `or` clause at line 2184-2185 returns False
when only ONE side is TyMemTier — the strict-separation
contract.

But the new cascade arm at 2178-2179 fires BEFORE the TyMemTier
arm. So `_compatible(TyVar('T'), TyMemTier(tier=Lt, inner=...))`
returns True via the cascade, bypassing the TyMemTier
strict-separation. The intent of cascade-safe ("mono will bind T
later") is reasonable: a generic T could legitimately be bound
to a memory-tiered type at the call boundary. But the TyMemTier
arm's docstring says "incompatible across tiers" — there is no
asterisk for "unless the other side is generic".

In practice, the AST constrains where each appears. TyVar
typically shows up inside TyFn.params or TyArray.elem of a
generic-fn signature. TyMemTier shows up in memory-tier-annotated
expressions. The interaction would happen at a call boundary
like:

```helix
fn f[T](x: T) { ... }
fn main() {
    let lt: Lt<i32> = ...;   // TyMemTier(tier=Lt, inner=i32)
    f(lt);                    // _compatible(TyVar('T'), TyMemTier(...))
}
```

The current arm returns True; mono substitutes T = TyMemTier(Lt, i32),
and the body of `f` then sees TyMemTier. Inside `f`, any use of
`x` against a non-tiered context (e.g., `x + 1`) would
re-check at the body via `_compatible(TyMemTier(...), TyPrim(i32))`
which now hits the TyMemTier `or` arm and returns False. So the
hole only opens IF mono actually allows the substitution and IF
the body uses x in a way that doesn't re-check. If the body uses
`x` only in tier-preserving ops, the program is well-typed.

The "weakness" is that the TyMemTier docstring's "incompatible
across tiers" is a sound contract only for non-generic positions.
The cascade arm silently relaxes it for generic-substitution
positions. This is theoretically correct (mono will defer the
real check) but a future reader would have to know the cascade
arm exists; the TyMemTier docstring doesn't note the carve-out.

The cycle-5 F1 recommended fix wording was: "Add a TyVar/TySize
cascade-safe arm at the top of `_compatible` (just below the
TyUnknown arm at line 2158-2159)". The cycle-6 fix placed it
exactly there. The recommendation did not anticipate the
TyMemTier interaction. The TyUnknown arm has the same
characteristic (TyUnknown vs TyMemTier returns True), so the
"top of function = cascade-safe for all" pattern is consistent —
this finding documents the trade-off rather than calling it a
bug.

**Reproducer**:
```python
from helixc.frontend.typecheck import (
    TypeChecker, TyVar, TyMemTier, TyPrim,
)
from helixc.frontend import ast_nodes as A
tc = TypeChecker(A.Program(span=A.Span(0,0), items=[]))
a = TyVar('T')
b = TyMemTier(tier='Lt', inner=TyPrim('i32'))
tc._compatible(a, b)
# -> True
# The TyMemTier arm at line 2182-2185 would return False
# if reached, but the cascade arm at 2178-2179 returns True first.
```

**Recommended fix**:
Option A (smallest diff, accept current behavior): document the
TyMemTier interaction in the cascade arm's comment. Update line
2169-2177 comment to add:

```python
# Note: this cascade runs BEFORE the TyMemTier strict-separation
# arm at line 2182-2185, so `_compatible(TyVar('T'),
# TyMemTier(...))` returns True. This is correct for the
# generic-substitution case — mono binds T to the tiered type and
# re-checks the body. The TyMemTier "no cross-tier mixing"
# contract therefore applies only to non-generic positions.
```

Option B (stricter, more invasive): move the cascade arm BELOW
the TyMemTier arm. Then `_compatible(TyVar('T'), TyMemTier(Lt,
i32))` returns False, and the call boundary forbids passing a
tiered value to a generic-T position. This is the "no implicit
generic-over-tier" reading — stricter but probably wrong: the
generic-fn-over-tier idiom is a useful pattern (e.g.
`fn id[T](x: T) -> T { x }` should accept any T including
tiered T).

Option C: split the cascade arm into two — TyVar cascades
unconditionally (it can bind any type), but TySize only cascades
when the other side is a size-shaped type (TySize, TyPrim('size_N'),
TyUnknown). This is a refinement of the type-kind separation
and probably overkill at this stage.

Option A is preferred — the current behavior is correct, the
fix is a documentation update only.

Severity rationale: no soundness bug (the behavior is correct for
the generic-substitution case). The "issue" is the implicit
relaxation of the TyMemTier docstring's stated contract. LOW.

---

## Cycle 5 finding re-verification

| ID  | Status     | Notes                                                                                                                                                                                                                                                                |
|-----|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| F1  | CLOSED     | `_compatible` now has a TyVar/TySize cascade arm at line 2178-2179. `_compatible(TySize('N'), TySize('M'))` returns True. The cycle-4 E1 primary reproducer (`fn f[N](a:[i32;N])` called from `fn g[M](a:[i32;M])`) is now closed. See G2 for an ordering observation. |
| F2  | CLOSED     | D-domain binop gate at line 1349-1371 is broadened to `inner_mismatch OR (l_is_diff != r_is_diff)`, symmetric with cycle-4 E2's Logic broadening. `D<f64> + f64` (same inner, asymmetric wrap) now warns with `"(one side D-wrapped, other bare)"`. See G1 for a diagnostic-text observation on the mixed-domain `D<f64> + Logic<f64>` case. |
| F3  | CLOSED     | `helixc/backend/x86_64.py:3032-3036` now does `if mono_diags: for d in mono_diags: print(f"error: fn-mono: {d}", ...); sys.exit(1)`. The driver aborts cleanly on any ShapeFoldError. Pipeline cannot continue into grad_pass / typecheck / codegen with a partial-mono program. |
| F4  | CLOSED     | `helixc/frontend/autodiff.py:709-721` now constructs a fresh `A.TileLit(span=..., dtype=..., shape=[_inline_lets(s, env) for s in expr.shape], memspace=_inline_lets(expr.memspace, env), init=...)` — walks both children as cycle-5 doc recommended. `let N = 4; tile<f32, [N], REG>::zeros()` substitutes correctly. |
| F5  | CLOSED*    | The cycle-5 F5 finding was about C4-2's BROADENING of tag-12 to Binary/Unary/Index/Field/If/Match/Block/UnsafeBlock RHS, making the namespace collision affect many more variables. Cycle-6 REVERTS that broadening — back to the cycle-2 D2 narrow Call-only sentinel. The cycle-5-F5 escalation surface is gone. The underlying D2 tag-12 vs AST_LET_MUT collision still exists (Call sentinel uses tag 12), but that was a pre-cycle-4 known issue documented in cycle-4 E5 + cycle-5 F5; it is NOT NEW in cycle 6 and is functionally narrower (one code path, not eight). Not re-flagged. |
| F6  | CLOSED     | The cycle-5 F6 finding was about the C4-2 AST_VAR arm's empty defer body (`if val_tag == 1 { /* defer */ }`). Cycle-6 REVERTS the entire C4-2 broadening block — the AST_VAR arm is gone. The original D2 Call-only sentinel doesn't have an AST_VAR arm at all (val_tag == 16 is Call), so the F6 surface is eliminated. |

All six cycle-5 findings are addressed at the contract level. The
two new cycle-6 findings (G1, G2) are about secondary surface
created or exposed by the F1 and F2 fixes — both LOW severity and
both diagnostic-quality rather than soundness.

---

## Cycle 6 invariant snapshot (post-fix)

The cycle-6 fix-sweep moved several invariants up a strength tier:

**`_compatible` contract** (typecheck.py:2166-2269):
- TyUnknown cascades (a or b is TyUnknown → True).
- **NEW (F1)**: TyVar/TySize cascades (a or b is TyVar/TySize → True).
  Defers to mono substitution. See G2 for an interaction note with
  the TyMemTier arm.
- TyMemTier: strict tier match required when both sides are TyMemTier;
  False if only one side is TyMemTier — EXCEPT when the other is
  TyVar/TySize/TyUnknown (cascade fires first).
- TyQuote / TyDiff / TyLogic / TyTuple / TyArray / TyRef / TyPtr /
  TyFn / TyTensor / TyTile: kind-tagged-equal arms that recurse on
  inner types. TyArray.size compare disjunctive `==` OR `_compatible`
  (cycle-4 E1).
- Catch-all: `return a == b` (identity).

**D-binop contract** (typecheck.py:1349-1410):
- Gate fires on `(l_is_diff or r_is_diff) AND (inner_mismatch OR
  (l_is_diff != r_is_diff))`. Closes asymmetric-wrap same-inner
  case (`D<T> + T`).
- **NEW interaction (G1)**: gate also fires on `D<T> + Logic<T>` and
  reverse — diagnostic extra text says "(one side D-wrapped, other
  bare)" which is inaccurate for the Logic-wrapped operand.

**Logic-binop contract** (typecheck.py:1372-1399):
- Gate fires on `(l_is_logic or r_is_logic) AND (inner_mismatch OR
  (l_is_logic != r_is_logic))`. Closes Logic asymmetric-wrap
  (cycle-4 E2). Only reached when D-arm doesn't fire (elif chain).

**TileLit AST walk contract** (autodiff.py:709-721):
- `_inline_lets(TileLit)` now recurses through `shape: list[Expr]`
  and `memspace: Expr`. `dtype: TyNode` is not walked (correct —
  TyNode is a type-annotation tree, not an Expr).

**fn-mono error severity** (x86_64.py:3022-3036):
- ShapeFoldError surfaces as `error: fn-mono: ...` and the driver
  exits 1. No silent-continue into the rest of the pipeline.

**Trap-id reader invariant** (typecheck.py + monomorphize.py):
- Every `TRAP_*` constant has at least one reader via f-string
  interpolation. The audit-time invariant from `docs/trap-ids.md`
  is satisfied.

**Parser closure-capture sentinel** (parser.hx:2330-2349):
- Reverted to the cycle-2 D2 narrow Call-only sentinel. Tag 12
  written for Call RHS only; Binary/Unary/Index/Field/If/Match/
  Block/UnsafeBlock RHS leave `inferred_ty_tag = -1` and the
  capture guard's `>0` check treats them as "untracked but assume
  i32" — known false-negative for non-i32 expressions, accepted
  in cycle 6 because the C4-2 broadening's false-positives on
  trivially-i32 Binary cases were worse than the false-negatives
  on the narrow Call-only path.

---

## Cycle 6 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **2 new findings (0 HIGH, 0 MEDIUM, 2 LOW)**.

By the strict criterion, **cycle 6 does NOT count clean**. However,
the cycle-6 fix-sweep is the cleanest yet:
- All six cycle-5 findings (F1-F6) are closed at the contract level.
- Zero HIGH or MEDIUM findings — the trend from cycle-1 (HIGH) →
  cycle-2 (HIGH+MED) → cycle-3 (MED) → cycle-4 (MED) → cycle-5 (3
  MED, 3 LOW) → cycle-6 (0 MED, 2 LOW) is monotonically
  decreasing in severity.
- Both G1 and G2 are diagnostic-quality issues on edges of the
  newly-broadened contract surfaces (F2's D-gate and F1's cascade
  arm), not soundness bugs.

The two LOW findings (G1, G2) follow the established pattern: each
cycle-N fix expands the contract surface, and the expanded surface
has a slightly-weaker secondary invariant than the cycle-N doc
described. This cycle's secondary invariants are about diagnostic
*text accuracy* (G1) and documentation of an *ordering interaction*
(G2) — not about silent-miscompile or invariant-violation paths.

Recommended fix sequence for cycle 7:

1. **G1**: refine the D-arm extra text to distinguish the
   Logic-wrapped vs bare other-side case (~5 lines at line 1364-
   1368). Apply symmetric refinement to the Logic-arm at 1394-1395
   for completeness (no behavioral change, just text).
2. **G2**: document the TyVar/TySize cascade vs TyMemTier
   interaction in the cascade arm's comment (line 2169-2177). No
   code change needed; the behavior is correct, only the
   docstring needs to note the carve-out for generic-substitution
   positions.

Both fixes are 1-commit, low-risk. The cycle-7 audit gate may
discover further secondary surface from G1's text-refinement or
G2's comment-only update — historically each cycle's fixes have
exposed ~1-2 new findings. A realistic projection: cycle 7 likely
yields 0-2 LOW findings, and cycles 8-12 converge to a clean sweep
under the strict criterion.

The 5-clean-cycles requirement (per cycle-5 doc's projection)
remains the binding constraint for Python-helixc deprecation. At
the current 2-LOW-per-cycle rate, cycle 6+1 = 7 audit cycles
from now (cycle 13) is the earliest realistic clean-sweep horizon
under the strict criterion. Alternatively, relaxing the criterion
to "0 HIGH + 0 MEDIUM" (treating LOW as documented-but-deferred)
would mark cycle 6 as the first clean cycle and put the
5-clean-cycles bar within ~5 more audit rounds.
