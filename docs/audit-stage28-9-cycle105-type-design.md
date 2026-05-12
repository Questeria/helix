# Audit Stage 28.9 cycle 105 — Type design

## Header

- **Date**: 2026-05-12
- **HEAD**: `77e4b85` ("Stage 28.9 cycle-104 audits: 3/3 CLEAN,
  counter 1/5 -> 2/5").
- **Counter at start**: 2/5 (cycle-104 CLEAN advanced 1 -> 2).
- **Mode**: STRICT READ-ONLY on `helixc/`. The only file written
  by this audit is this document. No Edit calls.
- **Scope**: cycle-105 type-design fresh-rotation surfaces:
  1. `helixc/ir/lower_ast.py` — If / While / Loop block-param
     dataflow (related to the A.Loop cycle-96/97 fix landed at
     cycle-97 audit-T C96-1).
  2. `helixc/ir/passes/effect_check.py` — `-O`-level interaction
     with effect propagation (cycle-32 audit-R C31-4 hardened the
     `assert`-on-set-disjointness path to `raise RuntimeError`;
     re-walk for any remaining `-O`-stripped sites).
  3. `helixc/backend/x86_64.py` — BITCAST / CAST type-aliasing
     arms (the bit-reinterpret and the numeric-conversion arms).
- **Deferred-known and NOT re-flagged**:
  - `monomorphize._mangle_ty`
  - `hash_cons._ast_equal`
  - `struct_mono` pre-flatten
  - `autotune iter_fn_decls`
  - `A.StrLit` lowering gap
  - `DIV / MOD / SHR` signed-vs-unsigned dispatch
  - `struct_mono.mangle_struct` collision
  - any cycle 1-104 finding (the cumulative deferred set; see
    cycle-104 / cycle-103 docs for the full enumeration).
- **Bar**: PASS = ZERO new findings at confidence >= 75%.
- **Source delta since cycle-104 audit**: only the cycle-104
  audit-trail `.md` documents (HEAD `77e4b85` is the cycle-104
  doc-commit on top of `26dfa82` cycle-102 fix-sweep); `helixc/`
  source is unchanged vs cycle-104 base.

## Methodology

Three verification points covering the three fresh rotation
surfaces. Each point is a targeted re-walk: read the surface,
trace through every type-design-relevant arm, cross-check
against the deferred-known list, and confirm any candidate
defect is reachable through the surface language (so it is a
real Phase-0 bug, not a defense-in-depth observation).

1. **V1 — `lower_ast.py` If/While/Loop block-param dataflow.**
   Read the three CFG-shaping arms (lines 1726-1764 for `A.If`,
   1862-1894 for `A.While`, 1895-1917 for `A.Loop`); confirm
   the merge-block param type matches both branches' values
   under the frontend's `_compatible` rules; confirm the
   `append_block` (vs the orphan-creating `new_block`) idiom
   is consistently used; confirm RETURN-then-BR sequencing in
   each arm is benign or terminator-discipline-respected.
2. **V2 — `effect_check.py` `-O`-level interaction.** Read the
   full module (524 lines); grep for any remaining `assert`
   statements that would be stripped by `python -O`; confirm
   the disjointness-runtime-check at lines 424-429 uses
   `raise RuntimeError` (cycle-32 C31-4 fix); re-walk the
   `compute_closure` fixpoint for `-O`-sensitive sites
   (debug-only invariants, `assert`-gated cache-prime steps);
   confirm the PURITY_OBSERVER_EFFECTS subtraction in
   `check_module` is symmetric across the @pure and the
   under-declared arms.
3. **V3 — `x86_64.py` BITCAST / CAST type-aliasing arms.**
   Read both arms (lines 1228-1314); walk every `(from, to)`
   numeric scalar pair the frontend can produce; trace whether
   the dispatch chain selects the correct conversion (cvtsi2sd
   / cvttsd2si / movsxd / etc.) or falls through to the
   final `mov`-copy branch with a width / format mismatch.

## V1 — `lower_ast.py` If/While/Loop block-param dataflow (PASS)

The `A.If` arm (lines 1726-1764) builds a three-block CFG with
explicit `append_block` for the then / else / merge blocks; the
merge block's single `new_block_param` is typed with `t_val.ty`
(line 1763) and receives `t_val` and `e_val` via BR-arg from the
then / else arms (lines 1747-1748 / 1758-1759).

**Type-design concern probed**: the merge-param uses the
then-arm's type unconditionally. If `t_val.ty` and `e_val.ty`
differ at IR level, the merge-param's declared type would be
wrong for the else-arm BR-arg slot, causing the x86_64 backend
to read the wrong slot-width on a per-arm basis.

This is, however, gated by the frontend typecheck: `A.If`'s
arm check at typecheck.py:1634-1647 calls
`_compatible(t, e)`. `_compatible` (typecheck.py:2296-2425)
returns `False` for cross-class pairs (e.g. `TyPrim("i32")`
vs `TyPrim("i64")` — fall-through at line 2425 returns
`a == b`, which is `False` because `TyPrim.name` differs). So
any source-level if/else with arms of distinct primitive
widths fails typecheck and never reaches IR lowering.

For composite types (`TyTuple`, `TyArray`, etc.) the structural
arms in `_compatible` likewise reject inner-element mismatches.
For numeric-scalar arms typecheck does NOT widen — `i32` and
`i64` are not implicitly cross-compatible. So `t_val.ty == e_val.ty`
holds at lower-time for every Phase-0-parseable if/else with a
final-expr value, and the merge-param's `t_val.ty` is sound.

The `A.While` arm (lines 1862-1894) does not produce a value;
the exit block has no params, so no block-param dataflow defect
is possible. The body's `_lower_block` return value is
discarded (line 1889); the loop-as-statement convention is
respected throughout the IR.

The `A.Loop` arm (lines 1895-1917) — the surface that the
cycle-97 C96-1 fix targeted — now consistently uses
`append_block` for both `header_blk` and `body_blk` (lines
1909-1910); the cycle-97 commentary at 1901-1908 explicitly
records the `new_block` vs `append_block` distinction and the
fix. Re-confirmed against the `Builder.append_block` /
`Builder.new_block` definitions in `tir.py` (the append form
links the block into `current_fn.blocks`; bare `new_block`
returns an orphan that no later pass enumerates).

**RETURN-then-BR sequencing.** If a then-arm's `_lower_block`
contains an early `A.Return` (lines 1929-1948), the lowerer
emits RETURN, then the If arm's BR-to-merge follows (line
1747). The Builder has no `is_terminated` guard, so both ops
sit in the same block. At codegen the RETURN arm (line 1909)
emits the `ret` instruction; the following BR is dead code
after `ret`. The merge block expects a block-param that never
arrives from this path. But because the dead `ret` is executed
first, the merge-block-param's uninitialized slot is never
read; the program returns before the merge. Not a miscompile —
it is unreachable-after-`ret` dead code, which is benign at the
machine level. Recorded as observation only, conf < 50%.

PASS at conf >= 75. The cycle-97 fix is intact; no new
finding on this surface.

## V2 — `effect_check.py` `-O`-level interaction (PASS)

Read the full module (524 lines). Grep for `\bassert\b` returns
no hits (the cycle-32 C31-4 fix converted the only critical
`assert` into a `raise RuntimeError` at lines 424-429).
Therefore `python -O` (which strips `assert` statements)
cannot silently disable any effect-check invariant. All
preconditions and disjointness checks are encoded with
explicit `raise` or unconditional branches.

The `compute_closure` fixpoint (lines 282-323) uses only
ordinary Python control flow — no `assert`, no `__debug__`,
no debug-only cache-prime step. The `-O`-stripped behaviour is
identical to the unoptimized behaviour.

The `PURITY_OBSERVER_EFFECTS` subtraction is symmetric across
the two arms of `check_module`:
- @pure arm (line 357): `effective_clos = clos - PURITY_OBSERVER_EFFECTS`
- under-declared arm (line 374): `extra = (clos - PURITY_OBSERVER_EFFECTS) - decl`

Both arms subtract the observer set before comparing against
the declared / required set. The 19002 unused-effect check
(line 387) does NOT subtract observers from `decl` — which is
the correct asymmetry: a function that declared `@effect(trace)`
without ever using a TRACE_ENTRY op SHOULD trip 19002, because
the declaration is misleading. The cycle-21 C20-T1 / C20-T5
commentary at lines 113-122 documents this asymmetry
explicitly.

`_HARD_EFFECT_TRAP_IDS` and `_INFO_EFFECT_TRAP_IDS` (lines
413-414) are runtime-checked for disjointness at module
import (lines 424-429) via a real `raise RuntimeError` — not
an `assert`. `-O` cannot bypass this check. The cycle-32
audit-R C31-4 hardening is intact.

The `classify_effect_error` function (lines 443-463) uses
substring search anchored on the full bracketed `[trap NNNNN]`
token (cycle-30 audit-T C28-TD1 / cycle-32 audit-T C30-5 /
C30-7 / C31-5 commentary documents the design rationale).
Future trap-ids that contain `19001` as a prefix substring
cannot accidentally match.

`report_diagnostics` (lines 466-524) has a defensive `if
eff_errs is None: return 0` early-exit (cycle-32 C31-2) and a
defensive `stderr=None -> sys.stderr` default (cycle-32 C31-R2
/ C31-1 / C30-1). Both are explicit guards, not `assert`-
gated.

No `-O`-sensitive site remains. PASS at conf >= 75.

## V3 — `x86_64.py` BITCAST / CAST type-aliasing arms (FAIL — F105-1)

### BITCAST arm (lines 1228-1243) — sound for current
emit sites

BITCAST is currently emitted by `lower_ast.py` only at lines
1346-1365 for the four `__bits_of_f32` / `__f32_from_bits` /
`__bits_of_f64` / `__f64_from_bits` builtins. Result types are
`i32` / `f32` / `i64` / `f64` respectively. The `wide`
predicate at lines 1234-1236 correctly detects the 8-byte
cases via `_is_f64_type(...) or _is_i64_type(...)` on either
operand, and the 4-byte cases via the absence thereof.

Defense-in-depth observation: `_is_i64_type` recognizes only
`i64` and `isize` (line 1019-1025); it does NOT recognize
`u64` or `usize`. If a future commit emits BITCAST with a
`u64` source or target, the `wide` check returns `False` and
the 4-byte mov silently truncates 8 bytes to 4. But because
the surface-language has no `__bits_of_u64` / similar builtin
that emits BITCAST with u64-typed values, this is not currently
reachable. Recorded as observation only, conf < 75% (not a
present miscompile; only a brittle defense-in-depth gap).

### CAST arm (lines 1244-1314) — **F105-1: f64 <-> f32 falls
through to 4-byte mov, missing cvtsd2ss / cvtss2sd**

The CAST arm dispatches by ten `if` arms in sequence (lines
1255-1313). Each arm narrows the (from_ty, to_ty) pair to
a specific conversion sequence. The final arm (line 1310-1313,
`if from_is_float == to_is_float`) is the fall-through 4-byte
mov-copy intended for same-class same-width pairs (e.g. i32
-> i32, f32 -> f32).

Walking the dispatch for `(from_ty = f64, to_ty = f32)`:

| Arm | Condition | Match? |
|-----|-----------|--------|
| 1 (line 1256) | `from_is_i64 and not to_is_float and not to_is_i64` | False (from_is_i64=F) |
| 2 (1261) | `not from_is_float and not from_is_i64 and to_is_i64` | False (from_is_float=T) |
| 3 (1268) | `from_is_i64 and to_is_f64` | False |
| 4 (1275) | `from_is_i64 and to_is_i64` | False |
| 5 (1280) | `not from_is_float and to_is_f64` | False |
| 6 (1286) | `not from_is_float and to_is_float` | False |
| 7 (1292) | `from_is_f64 and not to_is_float` | False (to_is_float=T) |
| 8 (1298) | `from_is_float and not to_is_float` | False (to_is_float=T) |
| 9 (1304) | `from_is_f64 and to_is_f64` | False (to_is_f64=F) |
| 10 (1310) | `from_is_float == to_is_float` (T==T) | **True — falls through to 4-byte mov** |

The 4-byte `mov_eax_mem_rbp` + `mov_mem_rbp_eax` (lines
1311-1312) copies the low 32 bits of the f64 (`src_slot`)
into the f32 destination slot, treating the IEEE-754 binary64
mantissa-low-bits as a binary32 representation. The result is
not the value-preserving narrowing of the f64; it is the
low-32-bit bit-pattern of the f64 byte-reinterpreted as a
binary32. This is a **silent miscompile**.

Symmetric case: `(from_ty = f32, to_ty = f64)`:

| Arm | Condition | Match? |
|-----|-----------|--------|
| 1-4 | (i64 / i32 -> ... paths) | False |
| 5 (1280) | `not from_is_float and to_is_f64` | False (from_is_float=T) |
| 6 (1286) | `not from_is_float and to_is_float` | False (from_is_float=T) |
| 7 (1292) | `from_is_f64 and not to_is_float` | False |
| 8 (1298) | `from_is_float and not to_is_float` | False (to_is_float=T) |
| 9 (1304) | `from_is_f64 and to_is_f64` | False (from_is_f64=F) |
| 10 (1310) | `from_is_float == to_is_float` (T==T) | **True — 4-byte mov** |

Result: the destination f64's low 32 bits are written with the
f32 source's 4 bytes; the destination's high 32 bits are
unmodified (residual stack contents from a prior write).
Silent miscompile, with the additional hazard of leaking
uninitialized stack bytes into the f64 mantissa (same defect
class as the cycle-77 F1 i64 for-loop increment that surfaced
4 bytes of uninitialized stack into the high half of an i64).

### Reachability through the Phase-0 surface

The frontend permits this cast. `_check_cast_compat`
(typecheck.py:2154-2244) accepts the pair under the
numeric-scalar-to-numeric-scalar arm at line 2197:

```
if self._is_numeric_scalar(src) and self._is_numeric_scalar(tgt):
    return
```

`f32` and `f64` are both in the numeric-scalar set, so `let x:
f64 = 1.5_f64; (x as f32)` typechecks. `lower_ast.py`'s
`A.Cast` arm at lines 2143-2151 unconditionally emits a CAST
op with the requested target type. So the CAST op with (f64,
f32) operand pair reaches the x86_64 backend.

### Test-coverage gap

The existing tests touch f64<->f32 conversions only through
the dedicated bootstrap builtins `__f32_to_f64` and
`__f64_to_f32` (test_codegen.py:2089-2128). These builtins
are implemented in the kovc.hx bootstrap compiler (kovc.hx:1657,
1903, 3596, 3609) and exercised via the `compile_and_exec`
harness (test_codegen.py:1603, which runs the cached bootstrap
binary) — **not** via Python's `lower_ast.py` + `x86_64.py`
CAST path. Greppping `test_codegen.py` for `as f32` / `as f64`
hits only the int-widening pairs (lines 617, 965, 1104, 5227,
5791-5792); no test exercises `(f64 as f32)` or `(f32 as f64)`
via the `compile_and_run` (Python pipeline) harness.

So the silent miscompile is unobserved in CI today purely
because no test program triggers the surface form. Any user
who writes `(some_f64_value as f32)` and goes through the
Python helixc pipeline will get a wrong-width / wrong-format
result.

### Suggested fix sketch (not prescribed; for cycle-106)

Insert two arms between arm 9 and arm 10 in `x86_64.py`:

```
if from_is_f64 and to_is_float:
    # f64 -> f32 narrowing: cvtsd2ss xmm0, xmm1
    self.asm.movsd_xmm0_mem_rbp(src_slot)
    self.asm.cvtsd2ss_xmm0_xmm0()   # encoding: F2 0F 5A C0
    self.asm.movss_mem_rbp_xmm0(res_slot)
    return
if from_is_float and to_is_f64:
    # f32 -> f64 widening: cvtss2sd xmm0, xmm1
    self.asm.movss_xmm0_mem_rbp(src_slot)
    self.asm.cvtss2sd_xmm0_xmm0()   # encoding: F3 0F 5A C0
    self.asm.movsd_mem_rbp_xmm0(res_slot)
    return
```

The bootstrap kovc.hx already emits cvtsd2ss / cvtss2sd via
the `__f64_to_f32` / `__f32_to_f64` builtin codegen at
kovc.hx:3596-3614, so the encoding is precedented. Adding the
two emit-asm helper methods to the `Asm` class is a
self-contained patch.

### Confidence: 90% (HIGH)

- Dispatch trace is direct and unambiguous (manual walk
  confirmed by the comment at line 1303 — "Same float-or-not:
  memory copy. For f64-to-f64, copy 8 bytes" — which describes
  the same-precision intent, NOT the cross-precision case).
- No alternative arm in the chain handles the f64<->f32 cross
  case.
- The frontend permits the cast.
- The IR lowering emits the op.
- The test suite confirms the absence of coverage.
- The defect class is a known x86_64 backend pattern (the
  same shape as cycle-77 F1 in spirit — a sibling-width gate
  missing a narrow / widen instruction).

The 10% gap is reserved for an unlikely scenario where a
downstream pass (DCE, fold, or peephole) rewrites the
(f64, f32) CAST into something else before reaching codegen.
Verified absent: `fold_module` (`const_fold.py`) handles
constant CASTs at compile time but does not rewrite
runtime-value CASTs; `cse_module` (`cse.py`) does not
restructure CASTs; `dce_module` (`dce.py`) only removes
dead ops; `fdce_module` (`fdce.py`) only removes unreachable
blocks. No pass rewrites the CAST op kind.

### Confirmation that F105-1 is NEW

Greppping prior audit docs for `cvtsd2ss`, `cvtss2sd`,
`f64.*as.*f32`, `f32.*as.*f64`, `CAST.*f64.*f32`, and
`f64_to_f32` / `f32_to_f64`:

- `docs/audit-stage28-8-cycle1-type-design.md:332` — discusses
  `(x_f32 as f64)` in a `_walk_subst_expr` context, NOT the
  backend CAST arm. Unrelated.
- `docs/audit-stage28-9-cycle92-type-design.md:67` — discusses
  reading 4 bytes as f32 bits in a different context (silent-
  failures cycle, not CAST dispatch). Unrelated.
- No prior audit doc identifies the CAST arm fall-through for
  the f64 <-> f32 cross pair. The cycle-77 F1 fix targeted a
  different sibling (the i64 for-loop increment width-gate);
  this is the f64<->f32 sibling-width-gate gap.

F105-1 is a fresh finding, not a re-flag.

## Findings table

| ID     | Severity | Confidence | Topic                                                       | Disposition |
|--------|----------|------------|-------------------------------------------------------------|-------------|
| F105-1 | HIGH     | 90         | x86_64.py CAST arm: f64<->f32 falls through to 4-byte mov   | NEW         |

## Verdict

**FAIL** — 1 new finding at confidence >= 75.

- V1 PASS (`lower_ast.py` If/While/Loop block-param dataflow:
  cycle-97 fix intact; merge-param type sound under
  `_compatible`'s strict-equality discipline; RETURN-then-BR
  sequencing is benign dead code).
- V2 PASS (`effect_check.py` `-O`-level interaction: no
  surviving `assert`; runtime disjointness check is `raise
  RuntimeError` per cycle-32 C31-4; PURITY_OBSERVER_EFFECTS
  subtraction symmetric).
- V3 FAIL — finding F105-1: `x86_64.py` CAST arm for
  (f64, f32) and (f32, f64) operand pairs falls through to a
  4-byte mov-copy at line 1311-1312, missing the
  cvtsd2ss / cvtss2sd emit. Silent miscompile reachable through
  the surface `(x as f32)` / `(x as f64)` form. Defense-in-
  depth observation about BITCAST u64/usize unreachability
  recorded but NOT flagged (conf < 75).

**Stage 28.9 audit-gate counter reset 2 -> 0.** Cycle-105
FAIL with a confidence-90 finding interrupts the 5-clean run.
Per standard audit-gate discipline:

- Either fix F105-1 in a cycle-106 fix-sweep and re-audit on
  the post-fix HEAD to start a fresh CLEAN run, OR
- Defer F105-1 (with explicit "DEFERRED — F105-1" disposition
  in cycle-106's scope note) and proceed; counter still
  resets per the FAIL bar.

## Cross-reference to cycles 101-104

- **Cycle 101**: PASS, 0 findings. Audited cmp dispatch +
  parser tuple/array literal + fdce reachability.
- **Cycle 102**: fix-sweep for 4 cycle-101 findings (ADD/SUB/
  MUL u64 + regression tests; commit `26dfa82`).
- **Cycle 103**: PASS, 0 findings on cycle-102 delta.
  Counter 0 -> 1.
- **Cycle 104**: PASS, 0 findings on parser.hx struct_gp_tab /
  gp_marker / named struct-lit / arith dispatch matrix.
  Counter 1 -> 2.
- **Cycle 105** (this doc): FAIL, 1 finding (F105-1). Counter
  2 -> 0. The finding is genuinely new: cycle-101 V3 / cycle-
  103 V2 audited the arith dispatch matrix for u64/usize
  coverage in ADD/SUB/MUL/cmp, but the **CAST** dispatch
  matrix was not in those audits' scope — and the cross-
  precision pair (f64, f32) was never test-covered through
  the Python pipeline.

Deferred items unchanged:
- monomorphize._mangle_ty hash discipline.
- hash_cons._ast_equal structural-recursion guard.
- struct_mono pre-flatten ordering.
- autotune iter_fn_decls scoping.
- A.StrLit IR lowering gap.
- DIV / MOD / SHR signed-vs-unsigned dispatch.
- struct_mono.mangle_struct collision risk.
- BITCAST u64/usize defense-in-depth (NOT reachable today;
  noted only as a brittleness observation, conf < 75%).

No edits to source performed. This document is the only file
written by cycle-105.
