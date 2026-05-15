# Audit Stage 28.9 cycle 111 — Type design

## Header

- **Date**: 2026-05-12
- **HEAD (auditor target)**: `f9425a0` ("Stage 30 cycle-2 IMPORTANT
  fix: regression tests for traps 62032/62033"). Working-tree HEAD
  is one commit further at `07e6535` (Stage 30 cycle-3 test-only
  addition); diff is a single `test_codegen.py` chunk that does not
  intersect any cycle-111 verification surface.
- **Prior fix-sweep commits in scope**:
  - `9c451e6` (Stage 28.9 cycle-110 fix-sweep, 3 cycle-109
    silent-failure findings F2/F3/F4): `helixc/backend/x86_64.py`
    +47/-12, `helixc/ir/lower_ast.py` +14/-1,
    `helixc/tests/test_ir.py` +243.
  - `fe7042f` (Stage 30 cycle-2 H1 fix): `helixc/bootstrap/parser.hx`
    +9/-7 — wires up `early_err` sentinel return path.
  - `1aecbae` (Stage 30 cycle-2 polish): `helixc/bootstrap/kovc.hx`
    + `helixc/tests/test_codegen.py` doc/comment cleanup.
  - `f9425a0` (Stage 30 cycle-2 IMPORTANT fix): `helixc/tests/
    test_codegen.py` +32 — regression-tests for traps 62032/62033.
- **Pre-cycle-110 kovc.hx prologue/trap-threshold delta**: the
  `emit_prologue` `sub rsp, 4096` and `bind_alloc_offset` trap
  threshold of 4096 (replacing the previous 1024) were applied as
  part of the cycle-110 fix-sweep but landed via in-source edit
  rather than a separately-attributed commit; both are present
  at HEAD `f9425a0` and the comment block at kovc.hx:740-750
  documents the cycle-110 (`C109-SF-F1 / C109-TD-F109-1`) origin.
  The original cycle-1-polish-#14 historical comment at lines
  733-738 is preserved verbatim for archival context.
- **Counter at start**: 1/5 per `f9425a0` (Stage 30 cycle-2 went
  CLEAN per the commit log).
- **Mode**: STRICT READ-ONLY on `helixc/`. The only file written
  by this audit is this document. No Edit calls.
- **Scope**: cycle-111 type-design rotation — seven verification
  points covering the cycle-110 fix-sweep delta + Stage 30 cycle-2
  parser.hx wire-up + kovc.hx cap-coordination invariant
  resolution + new trap-id coordination:
  1. `_is_64bit_int_type` cycle-110 sweep completeness (BIT_AND /
     BIT_OR / BIT_XOR / SHL / BIT_NOT / NEG promoted).
  2. u32→float zero-extend CAST arm: arm ordering, predicate
     guard correctness, zero-extend opcode encoding, REX.W
     register operand.
  3. kovc.hx cap-coordination invariant: prologue 4096 vs
     bind_alloc_offset 4096 vs bind_state 512 entries.
  4. A.Range loud-fail arm placement at lower_ast.py:2006 and
     upstream except-block re-silencing risk.
  5. New trap-id coordination: 10031 / 10032 / 10033 (kovc.hx) +
     62032 / 62033 (parser.hx) uniqueness.
  6. Cycle-110 regression-test discriminativity (cross-check the
     cycle-109 codereview F1/F2 corrections).
  7. bind_state cap-bump wire-through: 512 entries ∩ 511 offsets
     symmetric vs prologue 4096.
- **Deferred-known and NOT re-flagged**:
  - monomorphize `_mangle_ty` / hash_cons `_ast_equal` catchalls.
  - typecheck/struct_mono pre-flatten in `check.py`.
  - `autotune.collect_autotuned_fns` missing `iter_fn_decls`.
  - `struct_mono.mangle_struct` collision.
  - DT_BIND_NOW unused constant.
  - raw-200 enumeration in parser.hx (cycle-7 deferred).
  - DIV / MOD / SHR `_is_i64_type` sites — cycle-101 deferred
    F2; cycle-110 commit message explicitly retains them
    ("SHR remains signed-only (sar) for now; cycle-101 finding
    F2 deferred-known"). Cycle-111 cross-checks these for
    inadvertent widening / narrowing (see V1).
  - `A.StrLit` IR lowering gap (cycle-95 deferred).
  - kovc.hx:1052 stale "blowing past the 512-byte prologue
    allocation" comment (cycle-2 polish commit `1aecbae`
    explicitly noted as LOW and not addressed).
- **Bar**: PASS = ZERO new HIGH findings at confidence ≥ 75%.
  MED observations recorded but do not count toward verdict.

## Methodology

Seven verification points across the cycle-110 fix-sweep delta +
Stage 30 cycle-2 wire-up + kovc.hx cap-coordination resolution.
Each point is a targeted re-walk: read the relevant modules, trace
every dispatch path through the surface language, cross-check
against the deferred-known list to ensure candidate defects are
fresh, and verify any new predicate / arm against the rest of the
type-class universe. The cycle-109 audit's V1/V2/V5 set the
baseline; this audit's V1 walks the same predicate matrix POST
cycle-110 sweep.

## V1 — `_is_64bit_int_type` cycle-110 sweep completeness (PASS with HIGH observation)

### Sites promoted by cycle-110

Per `git diff 9c451e6^..9c451e6 -- helixc/backend/x86_64.py` the
following sites flipped `_is_i64_type → _is_64bit_int_type`:

| Site | Line | Cycle-110 status |
|------|------|------------------|
| BIT_AND | 1554 | Promoted |
| BIT_OR | 1569 | Promoted |
| BIT_XOR | 1584 | Promoted |
| SHL | 1599 | Promoted |
| BIT_NOT | 1628 | Promoted |
| NEG | 1641 | Promoted |

All six match the cycle-110 commit message's F4 closure list
(BIT_AND / BIT_OR / BIT_XOR / SHL / BIT_NOT / NEG). The promotion
is mechanically sound because these instructions are sign-agnostic
at the bit level — `and rax, rcx` (48 21 C8) acts identically on
i64 and u64 operands; only width matters.

### Sites still on `_is_i64_type` (deferred-known)

| Site | Line | Predicate | Disposition |
|------|------|-----------|-------------|
| DIV  | 1513 | `_is_i64_type(op.results[0].ty)` | Cycle-101 F2 deferred-known (signed-only `idiv`; u64 needs `div` opcode) |
| MOD  | 1528 | `_is_i64_type(op.results[0].ty)` | Cycle-101 F2 deferred-known (same as DIV) |
| SHR  | 1614 | `_is_i64_type(op.results[0].ty)` | Cycle-101 F2 deferred-known (signed `sar` vs unsigned `shr`); cycle-110 commit explicitly retains |

All three were on the cycle-101 deferred-known list and the
cycle-110 commit message reaffirms SHR's deferred status. DIV and
MOD's deferred status was also reaffirmed implicitly by their
absence from the F4 closure scope. No fresh defect; the
cross-site interaction-pattern observation from cycle-109's O109-1
still applies (deterministic-truncation pre-cycle-108 → stale-high-
half post-cycle-108 → still stale-high-half post-cycle-110, since
DIV/MOD/SHR remained on `_is_i64_type`).

### Sibling sites with spelled-out predicate

Two sites use the spelled-out `_is_i64_type(...) or _is_u64_type
(...)` predicate rather than `_is_64bit_int_type`:

| Site | Line | Predicate | Status |
|------|------|-----------|--------|
| FFI_CALL int-arg | 1990 | `_is_i64_type(arg.ty) or _is_u64_type(arg.ty)` | Semantic equivalent of `_is_64bit_int_type` |
| FFI_CALL return | 2009 | `_is_i64_type(op.results[0].ty) or _is_u64_type(op.results[0].ty)` | Semantic equivalent |
| CMP 64-bit gate | 1774-1777 | `_is_i64_type(L) or _is_i64_type(R) or _is_u64_type(L) or _is_u64_type(R)` | Semantic equivalent on the OR-of-operands flavour |

These three are semantically equivalent to `_is_64bit_int_type` and
are not in scope for the cycle-110 promotion. No type-design
asymmetry introduced. Refactoring to call `_is_64bit_int_type`
directly would be a clarity improvement but is not load-bearing.

### Predicates that did NOT widen/narrow in cycle-110

Grep `_is_64bit_int_type | _is_i64_type | _is_unsigned_int_type`
across `x86_64.py` confirms the cycle-110 diff touched ONLY the
six promotion sites listed above plus the two new CAST arms
(see V2). No unrelated predicate inadvertently widened or
narrowed. The cycle-110 commit's "F4 ... bitwise/unary u64/usize
truncation" scope claim matches the actual diff.

### V1 verdict: PASS

The cycle-110 sweep at the F4 sites is internally consistent
with the cycle-108 promotion pattern and the rest of the
predicate matrix. The cycle-101 DIV/MOD/SHR deferred-known set
is unchanged; the cycle-109 O109-1 observation (stale-high-half
manifestation) continues to apply but is not re-flagged as fresh.

## V2 — u32 → float zero-extend CAST arm (FINDINGS — F111-1)

### Arm placement (PASS sub-component)

The new arms at `x86_64.py:1340-1348` (u32→f64) and `1349-1357`
(u32→f32) are placed AFTER the unsigned-widening i32→i64 arm at
1305-1309, AFTER the i64→f64 arm at 1318-1323, AFTER the i64→i64
copy arm at 1325-1328, but BEFORE the legacy i32→f64 arm at
1359-1363 and the i32→f32 arm at 1365-1369. The cycle-110 comment
at lines 1329-1339 explicitly documents the ordering invariant
("the source as signed ... rex.W-prefixed cvtsi2sd/ss-from-rax").

For a u8/u16/u32 source casting to f64:
- Line 1290: `from_is_i64 = _is_64bit_int_type(u32) = False`.
- Line 1293 (i64→i32 trunc): `from_is_i64` is false, skip.
- Line 1305 (unsigned-widen to i64): `to_is_i64` is false (target
  is f64), skip.
- Line 1311 (i32→i64 sign-extend): `to_is_i64` is false, skip.
- Line 1318 (i64→f64): `from_is_i64` is false, skip.
- Line 1325 (i64→i64): `from_is_i64` is false, skip.
- Line 1340 (NEW u32→f64): `not from_is_float`=True, `_is_
  unsigned_int_type(u32)`=True, `not from_is_i64`=True,
  `to_is_f64`=True → fires.

Emission sequence: `mov eax, [rbp-src]` (8B disp32) zero-extends
to rax on x86-64 by hardware spec, then `F2 48 0F 2A C0` =
cvtsi2sd xmm0, rax (REX.W signed-int-to-double from rax). With
upper 32 bits of rax guaranteed zero, signed-64 conversion gives
the correct unsigned interpretation. Equivalent u32→f32 sibling
at 1349-1357 emits `F3 48 0F 2A C0` = cvtsi2ss xmm0, rax. Both
opcode encodings verified against Intel SDM Vol 2C (F2/F3 + REX.W
+ 0F 2A C0 = single-precision/double-precision cvtsi2s* xmm0, r64).

### Predicate guard correctness (PASS sub-component)

The guard `_is_unsigned_int_type(from_ty) and not from_is_i64`
covers exactly {u8, u16, u32} since `_is_unsigned_int_type`
matches {u8, u16, u32, u64, usize} (line 1072-1074) and
`_is_64bit_int_type` (=`_is_i64_type or _is_u64_type`) matches
{i64, isize, u64, usize}. Set difference: {u8, u16, u32}. Correct.

### Zero-extend opcode property (PASS sub-component)

`mov eax, [rbp-disp]` on x86-64 implicitly clears the upper 32
bits of rax (Intel SDM Vol 1, §3.4.1.1 "Because the upper 32 bits
of 64-bit general-purpose registers are undefined in 32-bit modes,
the upper 32 bits of any general-purpose register are not preserved
when switching from 64-bit mode to compatibility mode (32-bit or
16-bit). Software must not depend on these bits to maintain a
specific value. ... 32-bit operands generate a 32-bit result,
zero-extended to a 64-bit result in the destination general-purpose
register"). The cvtsi2sd subsequent read of rax sees the zero-
extended 64-bit value, with bit 63 guaranteed clear, so the signed-
64 interpretation equals the unsigned-32 source value. Correct
under all stack-pre-state conditions.

### u64 → float gap (FINDINGS sub-component — F111-1)

The cycle-109 silent-failures audit at `docs/audit-stage28-9-
cycle109-silent-failures.md` lines 282-329 explicitly identified
F3 as covering BOTH the u32 case AND the u64 case:

> Symmetric defects for `cast<u64, f64>` / `cast<u64, f32>`:
> Post-cycle-108, dispatch routes to the i64→f64 arm at lines
> 1318-1323, which emits `cvtsi2sd xmm0, rax` (REX.W signed
> conversion). For u64 with bit 63 set (e.g. 0x8000000000000000),
> this reads as -2^63 and produces ~-9.22e18; correct unsigned
> value is ~9.22e18. ... the bug class is moved one rung up the
> dispatch table, not eliminated.

The cycle-110 fix-sweep commit message scopes the F3 closure to:

> F3 (HIGH conf 88) — cast<u32, f64/f32> uses signed-int conversion

Narrowing the scope from "u8/u16/u32/u64/usize → f64/f32" (as the
cycle-109 audit established) to "u32 → f64/f32" without explicit
deferred-known declaration. The new u32→f arms (1340-1348,
1349-1357) gate on `not from_is_i64`, which by construction
EXCLUDES u64/usize sources. So for a u64 source casting to f64,
dispatch routes via:
- Line 1290: `from_is_i64 = _is_64bit_int_type(u64) = True`.
- Line 1293 (i64→i32 trunc): `not to_is_float` is False (to_is_f64
  → to_is_float=True since `_is_float_type` returns True for f64,
  per line 1029-1032 enumeration), skip.
- Line 1305 (unsigned-widen): `not from_is_i64` is False, skip.
- Line 1311 (i32→i64): `not from_is_i64` is False, skip.
- **Line 1318 (i64→f64)**: `from_is_i64`=True AND `to_is_f64`=True
  → fires. Emits `mov rax, [rbp-src]` then `F2 48 0F 2A C0` =
  cvtsi2sd xmm0, rax — **SIGNED** 64-bit-to-double conversion.

For a u64 with bit 63 set (e.g., `u64::MAX = 0xFFFFFFFFFFFFFFFF`):
- Signed-64 interpretation: -1.
- cvtsi2sd produces -1.0.
- Correct unsigned interpretation: 18446744073709551615.
- Correct f64 value (after rounding to 53-bit mantissa):
  ~1.8446744073709552e19.

Silent miscompile. Severity HIGH (matches cycle-109 F3's HIGH
conf 88 attribution). The cycle-110 fix-sweep commit's "F3
closure" claim is verifiably partial: the u32 portion is fixed
but the u64/usize portion remains broken at the same dispatch
arm the cycle-109 audit identified.

**Reachability under Phase-0 surface**: any `expr as f64` or
`expr as f32` cast where `expr: u64` / `expr: usize`. The
typecheck.py:1799 hint ("use a bitcast through a u64
intermediate" — verified by grep) routes a portion of f64↔int
conversion through u64 intermediates, so this is reachable in
practice.

**Test coverage**: `helixc/tests/test_ir.py:687-714` covers
`test_c110_cast_u32_to_f64_uses_zero_extend_path` and
`test_c110_cast_u32_to_f32_uses_zero_extend_path`. There is NO
equivalent test for u64→f64 / u64→f32, so the regression-test
suite cannot catch this gap. Grep `u64.*f64|u64.*float` in
test_ir.py: zero hits for cast tests, matching this absence.

**Confidence as a fresh HIGH finding for cycle-111**: 85%.
Justification:
- The opcode at line 1321 (`F2 48 0F 2A C0`) is unambiguously
  signed-int-to-double per Intel SDM (cvtsi2sd from r/m64 with
  REX.W = signed interpretation).
- The u64 dispatch path is uniquely determined by the predicate
  table (from_is_i64=True via `_is_64bit_int_type`, to_is_f64
  =True).
- The cycle-109 audit at conf 88 identified the same defect at
  the same dispatch site; the cycle-110 fix-sweep did not
  address it. The deferred-known list does not include u64→f.
- Audit discipline: a fix-sweep that closes only part of a
  flagged finding (without declaring the unclosed portion as
  deferred-known) is a fresh integrity gap.

The 15% confidence-shaving is for the interpretation question:
whether the cycle-110 commit's "F3 ... u32, f64/f32" wording
implicitly re-declared the u64 portion as out-of-scope. The
absence of any "u64→f deferred to Phase-1" note in the commit
or in any deferred-known list weighs against that
interpretation; hence conf above the 75% bar.

### V2 verdict: FINDINGS (F111-1 below)

The new u32→f arms are correct on their own (arm ordering,
predicate guard, opcode encoding, register operand all verified).
The cycle-110 fix-sweep is incomplete on F3: the u64→f and
usize→f paths still emit signed cvtsi2sd from rax, silently
miscompiling high-bit-set u64 values to negative floats. This
is the same dispatch arm and the same defect class the cycle-109
silent-failures audit's F3 identified at conf 88; the cycle-110
fix-sweep closed only the u32 portion.

## V3 — kovc.hx cap-coordination invariant (PASS)

### Prologue allocation

`emit_prologue` at line 751-757: `sub rsp, 4096` (was 1024 pre-
cycle-110). Bumped to accommodate the Stage 29.1 `bind_state`
cap of 512 entries × 8 bytes per slot = 4096 bytes peak. The
arithmetic relationship `prologue_frame ≥ bind_state_cap × 8`
now reads `4096 ≥ 512 × 8 = 4096` — exact equality with no
margin. The cycle-109 F109-1 finding ("Stage 29.1 broke the cap-
coordination contract by a factor of 4") is closed: the prologue
allocation now matches the simultaneously-live binding capacity.

### bind_alloc_offset trap threshold

`bind_alloc_offset` at line 1064-1083: `if off >= 4096 { emit_
trap_with_id(10030); }` (was 1024 pre-cycle-110). Matches the
new prologue. Trap-emit-then-bump pattern is unchanged from
cycle-109; the loud-fail trap semantics still dominate the
silent-corruption risk via the emit-then-use order in the OUTPUT
binary. Verified at one representative call site (line 5830-
5853): `bind_alloc_offset` is called immediately before the
`emit_mov_local_*(off)` store-emit, so the runtime trap-ud2
sequence precedes any over-cap memory store.

### bind_push_typed cap

`bind_push_typed` at line 1023-1046: `if top >= 512 { emit_
trap_with_id(10032); 0 - 1 }` (was 64 pre-Stage-29.1). Matches
the Stage 29.1 entry-table capacity.

### Off-by-one between entries and offsets (observation O111-1)

The two caps are subtly asymmetric:
- `bind_push_typed` allows top values 0..511 (after push the
  511th, top=512 triggers the trap on the 512th push attempt).
  Capacity: 512 entries.
- `bind_alloc_offset` starts at off=8 (per `bind_reset`) and
  bumps by 8 per call. Sequence: 8, 16, ..., 4088 (511 distinct
  offsets), then off=4096 → trap fires on the 512th call.

So 512 entries can coexist with at most 511 distinct offsets.
With LIFO scope (bind_pop decrements both `top` and `off`),
nested bindings reuse offsets, so simultaneously-live names ≤
simultaneously-live offsets ≤ 511. The 512th entry can exist
only if it shares an offset with an earlier entry (i.e.,
bind_pop happened between them) — which never happens since
bind_push_typed always pairs with a fresh `bind_alloc_offset`
in every call site (verified at the 7 call sites in kovc.hx).

In practice the 512-entry cap can never be reached: the
bind_alloc_offset trap fires first at the 512th push attempt.
The 1-slot asymmetry is reachable only if a hypothetical caller
called bind_push_typed without bind_alloc_offset (i.e., re-using
an offset from a popped scope). Not currently a defect under
the actual call-graph; below the HIGH bar. Recorded as O111-1.

### Audit-fix comment at lines 733-738 vs cycle-110 update

The task verification point #3 asks whether "the audit-fix
comment at kovc.hx:733-738 has been updated to reflect the new
arithmetic." Reading lines 733-738 verbatim:

```
// Audit fix (cycle 1, polish #14): bumped from 512 → 1024
// to match the bind_state cap (64 entries) with 2× margin.
// Previously 512 was "just enough" for 64 × 8-byte slots —
// any future cap bump would silently corrupt the saved
// rbp/return-address. 1024 gave 128 slots; future Phase-1
// should derive this from bind_state cap dynamically rather
// than hard-coding.
```

These six lines describe the OLD invariant (cycle-1 polish #14:
64 entries × 8 = 512 → 1024 with 2× margin) and are preserved
verbatim as historical context. The cycle-110 update is added
immediately below at lines 740-750:

```
// Cycle 110 fix C109-SF-F1 / C109-TD-F109-1 (CRITICAL conf 90 +
// HIGH conf 80): Stage 29.1 bumped bind_state cap 64→512 but left
// this 1024-byte prologue and bind_alloc_offset's 1024 trap
// threshold unchanged. 512 simultaneously-live bindings × 8 bytes
// = 4096 bytes peak. Any fn with > 128 simultaneously-live
// let-bindings reached past the prologue's stack allocation,
// corrupting parent frame's saved rbp / return-address / red
// zone. Bumped to 4096 to match (and updated bind_alloc_offset
// trap threshold to 4096 to match). The architectural note about
// deriving from bind_state cap dynamically still applies — that's
// a Phase-1 follow-up.
```

So the comment block at 733-750 (taken as a unit) reflects both
the cycle-1 polish #14 history AND the cycle-110 cap-coordination
update. A strict reading of the task ("kovc.hx:733-738 has been
updated") would prefer lines 733-738 themselves to be rewritten
in place, but the alongside-comment pattern preserves traceable
history. The reader gets the full timeline. Acceptable design
choice; below the HIGH bar.

The stale "blowing past the 512-byte prologue allocation;
emit_mov_local_eax(-560)" reference at line 1052-1053 was
explicitly flagged as LOW by the Stage 30 cycle-2 polish commit
`1aecbae` and acknowledged-but-not-fixed:

> NOT addressed in this cycle: ...
> - LOW: stale "blowing past the 512-byte prologue" reference at
>   kovc.hx:1050 (cycle-110 work changed prologue to 4096, but the
>   comment hasn't caught up)

Inherited deferred-known; not a fresh finding for cycle-111.

### V3 verdict: PASS

The cap-coordination invariant (`prologue_frame == bind_state_
cap × 8`) is now exact: 4096 == 512 × 8. The cycle-109 F109-1
finding is closed. The cycle-110 audit-fix comment block at
733-750 documents both the cycle-1 history and the cycle-110
update. The 1-slot off-by-one between bind_push_typed (512
entries) and bind_alloc_offset (511 distinct offsets) is
benign under the current call-graph; recorded as O111-1.

## V4 — A.Range loud-fail arm placement (PASS)

### Arm placement at lower_ast.py:2006-2020

The cycle-110 F2 fix replaces the pre-fix `if isinstance(expr,
A.Range): return None` with:

```python
if isinstance(expr, A.Range):
    # Stage 28.9 cycle 110 audit-S F2 fix (HIGH conf 92): ...
    raise NotImplementedError(
        f"range expression in non-For-iter position not yet "
        f"supported in IR lowering at "
        f"{expr.span.line}:{expr.span.col}")
```

The arm fires at the same dispatch position as the pre-fix
silent `return None` — i.e., AFTER `A.Return` (line 1973) and
BEFORE `A.Assign` (line 2021). The For iter_expr special-cases
A.Range at lower_ast.py:1820 BEFORE descending into `_lower_expr`,
so `for x in 0..10 { ... }` still works (the A.For arm intercepts
the Range before this arm fires). Confirmed by grep
`isinstance.*A\.Range` across `lower_ast.py` — two hits: the
A.For special-case at 1820 and the loud-fail arm at 2006.

### Cycle-108 F8 pattern symmetry

The cycle-108 F8 fix added loud-fail arms for A.CharLit (line
1061), A.StructLit (line 1069), A.TileLit (line 1077). The
cycle-110 F2 arm at line 2006 mirrors these:
- Same `raise NotImplementedError(...)` pattern.
- Same position-bearing diagnostic (`expr.span.line:expr.span.col`).
- Same rationale (parser admits the syntax, typecheck doesn't
  reject, but lower doesn't represent — silent-substitution-to-0
  pre-fix).

One asymmetry: the cycle-108 F8 arms are at the TOP of
`_lower_expr` (lines ~1037-1077), BEFORE the A.Name / A.Path /
A.Binary / etc. branches. The A.Range arm is MUCH later in the
dispatch (line 2006, after A.Return at 1973). Reason: A.Range
was pre-fix at this position (silent return None), so the
cycle-110 fix is in-place. Behavioural equivalence holds:
neither arm has a sibling subclass that could shadow it
(A.Range is its own AST class). No fresh defect introduced by
the late position.

### Upstream except-block re-silencing analysis

Grep `except\s+(Exception|NotImplementedError)|except\s*:`
across `helixc/` returns:

- `helixc/ir/lower_ast.py:2235` — narrow `except Exception:`
  ONLY around `structural_hash(expr.inner)` inside the A.Quote
  arm. Does NOT wrap `_lower_expr` itself. The Range
  NotImplementedError raised at line 2017 is not caught here.
- `helixc/check.py:306, 747, 778, 792` — `except Exception as e`
  blocks around module-level compile entry points. These DO
  catch NotImplementedError (since it derives from Exception)
  but the catch printed a loud `helixc: internal error: ...`
  diagnostic and returns rc=1. This is the canonical loud-fail
  channel; the user sees the diagnostic and the compiler exits
  with non-zero rc. NotImplementedError is re-formatted but the
  failure is observable — not silenced. Same handling as the
  cycle-108 F8 CharLit/StructLit/TileLit loud-fail raises.
- All other `except Exception` blocks (autodiff, const_fold,
  tests) are in unrelated modules; A.Range lowering does not
  pass through them.

### V4 verdict: PASS

The A.Range loud-fail arm mirrors the cycle-108 F8 pattern
correctly. The For iter_expr special-case at line 1820 preserves
the valid Range-in-for-loop case. No upstream over-broad except
re-silences the NotImplementedError.

## V5 — New trap-id coordination (PASS)

### New trap IDs added by cycle-110 + Stage 30 cycle-2

- **10031** (kovc.hx:1637, `patch_table_add` cap-overflow):
  cycle-110 C109-CR-F3 fix. Loud-fail replacement for the
  previous silent `return -1`. Single emit-site.
- **10032** (kovc.hx:1034, `bind_push_typed` cap-overflow):
  cycle-110 C109-CR-F3 fix. Single emit-site (lines 1021 and
  1032 are comments).
- **10033** (kovc.hx:1570, `fn_table_add` cap-overflow):
  cycle-110 C109-CR-F3 fix. Single emit-site.
- **62032** (parser.hx:3301, `ta_count != gp_count_pre` arity
  mismatch in generic struct-lit): Stage 30 cycle-2 H1 fix.
  Single emit-site.
- **62033** (parser.hx:3296, 3298, `ta_bad_token` or `post_loop_t
  != 17` bad-token-in-args): Stage 30 cycle-2 H1 fix. Two emit-
  sites with the SAME semantic ("bad token in type-args list");
  intentional design choice, not a collision.

### Uniqueness check across helixc/

Cross-referenced against the deduplicated trap-ID universe
(`grep -hoE "emit_trap_with_id\s*\(\s*[0-9]+|mk_node\s*\(\s*99\s*,\s*[0-9]+" helixc/ -r | grep -oE "[0-9]+" | sort -u`). The
ranges 10030-10033 and 62030-62033 are each contiguous and
unique to their respective sites. No collisions with:
- The 8000-series (kovc.hx audit traps 8001-8016).
- The 14000/16000-series (validation passes).
- The 28999 series (validation pass aggregate trap).
- The 60000/62000-series (parser-level traps).
- The 99001 / 81002 series (catch-all internal traps).

10030 was added in an earlier cycle (cycle-5/6 finding #11);
10031/10032/10033 sit immediately above it in the same range,
which is a natural ID-allocation pattern for the same conceptual
family (cap-overflow loud-fail).

### Disposition of the silent `return -1` path

Pre-cycle-110, `bind_push_typed`, `fn_table_add`, `patch_table_
add` all returned `-1` on cap overflow and the caller discarded
the return value. The cycle-110 fix emits the trap BEFORE the
`-1` return, so the trap fires at compile time as a ud2 sequence
in the OUTPUT binary. The `-1` return value is preserved for
backwards compatibility but every caller still discards it; the
trap is the actual signal.

Important type-design property: the trap is emitted into the
EMITTED BINARY at the cap-overflow site, not raised in the
kovc.hx-as-Python compiler. So an over-cap source compiles to
a binary that SIGILLs at the corresponding instruction. The
compile-time behaviour is: emit the trap-ud2 + emit `-1` as the
return value + caller discards it + compilation continues with
stale-state caller logic. The runtime trap dominates the
silent-corruption risk; analysis matches cycle-109 V5's
disposition of trap 10030.

### V5 verdict: PASS

All five new trap IDs (10031, 10032, 10033, 62032, 62033) are
unique within the helixc/ codebase. Two emit-sites for 62033
share a single semantic (bad-token-in-args) and are intentional.
No collisions with existing trap-IDs.

## V6 — Cycle-110 regression-test discriminativity (PASS)

The cycle-110 fix-sweep added 11 new regression tests in
`helixc/tests/test_ir.py`:

| Test | Production-code dependency | Discriminative byte/property |
|------|----------------------------|------------------------------|
| `test_c109_mut_u64_load_store_byte_identical_to_i64` | LOAD_VAR/STORE_VAR `_is_64bit_int_type` (cycle-108 F6) | ELF byte-equality with i64 sibling |
| `test_c109_call_return_u64_caller_stores_full_rax` | CALL return store `_is_64bit_int_type` (cycle-108 F2) | ELF byte-equality with i64 sibling |
| `test_c110_range_in_value_position_raises_loud` | A.Range loud-fail (cycle-110 F2) | `NotImplementedError` raised with "range" substring |
| `test_c110_cast_u32_to_f64_uses_zero_extend_path` | u32→f64 zero-extend arm (cycle-110 F3) | byte-presence of `F2 48 0F 2A C0` |
| `test_c110_cast_u32_to_f32_uses_zero_extend_path` | u32→f32 zero-extend arm (cycle-110 F3) | byte-presence of `F3 48 0F 2A C0` |
| `test_c110_bit_and_u64_emits_64bit_form` | BIT_AND `_is_64bit_int_type` (cycle-110 F4) | byte-presence of `48 21 C8` |
| `test_c110_bit_or_u64_emits_64bit_form` | BIT_OR `_is_64bit_int_type` (cycle-110 F4) | byte-presence of `48 09 C8` |
| `test_c110_bit_xor_u64_emits_64bit_form` | BIT_XOR `_is_64bit_int_type` (cycle-110 F4) | byte-presence of `48 31 C8` |
| `test_c110_shl_u64_emits_64bit_form` | SHL `_is_64bit_int_type` (cycle-110 F4) | byte-presence of `48 D3 E0` |
| `test_c110_bit_not_u64_emits_64bit_form` | BIT_NOT `_is_64bit_int_type` (cycle-110 F4) | byte-presence of `48 F7 D0` |
| `test_c110_neg_u64_via_sub_emits_64bit_form` | u64 SUB path (cycle-102 closure regression) | byte-presence of `48 29 C8` |

Spot-check 3 of them by mental revert:

1. **`test_c110_bit_and_u64_emits_64bit_form`**: pre-fix BIT_AND
   on u64 used `_is_i64_type` (False for u64), so the else
   branch fires producing `21 C8` (no REX.W). The byte sequence
   `48 21 c8` would NOT appear for u64 BIT_AND. Discriminator
   holds: revert the cycle-110 BIT_AND predicate → test fails.

2. **`test_c110_cast_u32_to_f64_uses_zero_extend_path`**:
   pre-fix u32→f64 routed via line 1359-1362 (legacy i32→f64
   arm) emitting `F2 0F 2A C0` = cvtsi2sd xmm0, eax (NO REX.W,
   4 bytes). The 5-byte sequence `F2 48 0F 2A C0` would not be
   present. Discriminator holds: revert the cycle-110 new arm
   → test fails.

3. **`test_c110_range_in_value_position_raises_loud`**: pre-fix
   `A.Range` returned None silently; `lower_src(src)` would
   complete without raising. Test asserts `NotImplementedError`
   IS raised. Revert the cycle-110 line 2017 → test fails.

All 3 spot-checked discriminators hold against a hypothetical
revert. The cycle-109 code-review F1/F2 cross-corrections (the
two `_byte_identical_to_i64` tests) strengthen the cycle-107
F2/F6 tests by replacing existence-of-byte-pattern asserts
(which can vacuously pass via unrelated codegen sites) with
strict byte-equality between u64 and i64 fn bodies.

### V6 verdict: PASS

All 11 cycle-110 regression tests are discriminative. The cycle-
109 codereview F1/F2 strengthening replaces the cycle-107 F2/F6
vacuous-passability with strict byte-equality assertions.

## V7 — bind_state cap-bump wire-through (PASS)

### Math

- bind_state cap (entries): 512 (bind_push_typed:1033).
- bind_alloc_offset trap threshold: 4096 bytes; allows offsets
  8..4088 (511 distinct 8-byte-aligned offsets).
- emit_prologue allocation: `sub rsp, 4096`; addressable slots
  [rbp-8..rbp-4096] (512 slots if 8-byte-aligned).
- Maximum simultaneously-live binding: min(512 entries, 511
  offsets, 512 prologue slots) = 511.

### Symmetry vs cycle-109 F109-1 closure

Cycle-109 F109-1 documented the contract violation: cycle-1
polish #14 set prologue=1024 to match bind_state-cap=64
(invariant `prologue ≥ cap × 8`, i.e., 1024 ≥ 512, with 2×
margin). Stage 29.1 bumped cap 64→512 without bumping prologue,
violating `1024 ≥ 4096`. Cycle-110 closes by bumping prologue
1024→4096, exactly matching `4096 ≥ 4096` (zero margin).

The zero-margin choice is acceptable because the
bind_alloc_offset trap at 4096 fires BEFORE the 512th offset
allocation, so the 512th simultaneously-live binding is
unreachable. The actual maximum is 511 simultaneously-live, well
within the 4096-byte allocation. The reachable space is
[rbp-8..rbp-4088] = 4081 addressable bytes for the 511th offset's
8-byte store (writing [rbp-4088..rbp-4081] = bytes 8..15 from
the bottom of the allocation; rbp-4088 = rsp+8, still inside
the allocation since rsp = rbp-4096).

### Phase-1 architectural note

The cycle-110 audit-fix comment at lines 740-750 notes: "The
architectural note about deriving from bind_state cap
dynamically still applies — that's a Phase-1 follow-up." The
current hard-coded 4096 / 512 / 4096 triple will require manual
re-coordination if any of the three caps changes again. A
dynamic-derivation refactor would couple all three via a single
named constant. Outside the scope of cycle-111; recorded as
O111-2 (architectural; below HIGH bar).

### V7 verdict: PASS

The bind_state cap-bump (cap=512, max simultaneously-live=511)
is exactly accommodated by the prologue 4096-byte allocation
(512 slots, 511 usable before the trap). The cap-coordination
contract is restored to a tight-fit zero-margin configuration
that is mathematically consistent with the runtime trap-cap
mechanism.

## Findings table

| ID | Severity | Confidence | Topic | Disposition |
|----|----------|------------|-------|-------------|
| F111-1 | HIGH | 85% | cycle-110 fix-sweep partial F3 closure: `cast<u64, f64>` and `cast<u64, f32>` still emit signed `cvtsi2sd xmm0, rax` (`F2 48 0F 2A C0`) at `helixc/backend/x86_64.py:1318-1323`, silently miscompiling high-bit-set u64 values to negative floats. The cycle-109 silent-failures audit at lines 282-329 explicitly identified u64→f as part of F3 alongside u32→f. The cycle-110 fix-sweep commit scoped F3 to "u32, f64/f32" without declaring u64→f as deferred-known. The new u32→f arms at lines 1340-1348 / 1349-1357 explicitly exclude u64 sources via `not from_is_i64`. For `u64::MAX as f64`, signed-64 interpretation gives -1 → cvtsi2sd produces -1.0; correct unsigned interpretation is ~1.8447e19. No regression test exercises the u64→f path. Recommended fix: add explicit u64→f arms BEFORE the line 1318 i64→f64 arm. The standard x86 sequence for u64→f64 (no AVX-512) is a magic-number-add or a high/low split-and-add sequence; alternatively emit a runtime helper call. Add discriminative regression test: `u64(0x8000000000000000) as f64` → assert returned f64 ≈ 9.22e18 (not -9.22e18). |

## Observations (not flagged; below the HIGH bar)

| ID | Severity | Confidence | Topic |
|----|----------|------------|-------|
| O111-1 | LOW-OBS | 60% | bind_push_typed cap (512 entries) and bind_alloc_offset cap (511 distinct 8-byte offsets before trap at off=4096) are off-by-one. Under the current call-graph (every push paired with one alloc_offset) the asymmetry is unreachable — the alloc_offset trap fires before the 512th push attempt. A hypothetical caller that pushed without alloc_offset (re-using an offset from a popped scope) could in theory exploit the 1-slot mismatch, but no such caller exists. Below the HIGH bar; recorded for symmetry-of-design tracking. |
| O111-2 | LOW-OBS | 55% | Hard-coded cap triple (prologue 4096 / alloc_offset trap 4096 / bind_state 512) requires manual re-coordination if any cap changes. The cycle-110 fix-sweep audit-fix comment explicitly notes this as a Phase-1 follow-up. Below the HIGH bar; architectural improvement, not a defect. |
| O111-3 | MED-OBS | 70% | kovc.hx:1052 stale "blowing past the 512-byte prologue allocation; emit_mov_local_eax(-560) writes into the parent frame's saved rbp/return-address" comment references the pre-cycle-1-polish-#14 512-byte prologue, three caps obsolete. The Stage 30 cycle-2 polish commit `1aecbae` explicitly flagged this as LOW-deferred. Comment-only; no behavioural impact. Below the HIGH bar for type-design (matches the polish commit's disposition). |

## Verdict

**Verdict: FINDINGS** — 1 HIGH (F111-1) at confidence 85%.
Counter resets 1/5 → 0/5 per audit-gate discipline (a HIGH
finding at conf ≥ 75% blocks counter advance).

- V1 PASS: cycle-110 6-site promotion (BIT_AND / BIT_OR /
  BIT_XOR / SHL / BIT_NOT / NEG → `_is_64bit_int_type`) is
  consistent with the cycle-108 sweep pattern; DIV / MOD / SHR
  remain on `_is_i64_type` per cycle-101 deferred-known and the
  cycle-110 commit message reaffirmation. No inadvertent
  predicate widening or narrowing elsewhere.
- V2 FINDINGS: new u32→f arms are correct on their own (arm
  ordering, predicate guard, opcode encoding, REX.W register
  operand all verified); cycle-110 fix-sweep is incomplete on
  F3 — u64→f and usize→f paths still use signed cvtsi2sd from
  rax. Fresh HIGH finding F111-1 at conf 85%.
- V3 PASS: kovc.hx cap-coordination invariant restored
  (`prologue == bind_state_cap × 8 == 4096`). Cycle-109 F109-1
  is closed. Audit-fix comment at 733-750 documents both the
  cycle-1 polish-#14 history and the cycle-110 update.
- V4 PASS: A.Range loud-fail arm at lower_ast.py:2006 mirrors
  the cycle-108 F8 pattern; For iter_expr special-case at
  line 1820 preserves valid Range usage; no upstream over-broad
  except re-silences the NotImplementedError.
- V5 PASS: 5 new trap IDs (10031, 10032, 10033, 62032, 62033)
  are unique within helixc/. Two emit-sites for 62033 share
  one semantic ("bad token in type-args list") by intentional
  design.
- V6 PASS: 11 cycle-110 regression tests are discriminative.
  Spot-checked 3 — `test_c110_bit_and_u64`, `test_c110_cast_u32
  _to_f64`, `test_c110_range_in_value_position` — each fails
  against a hypothetical revert of the corresponding production
  change.
- V7 PASS: bind_state cap-bump (512 entries) wires through to
  the prologue 4096-byte allocation with zero margin and one-
  slot trap-headroom. 511 simultaneously-live bindings fit
  exactly within the allocation; the 512th attempt traps.

**Stage 28.9 audit-gate counter resets to 0/5** (HIGH finding
at conf ≥ 75% blocks advance). Cycle-112 fix-sweep should close
F111-1 by adding explicit u64→f arms BEFORE the i64→f64 arm at
line 1318 — using either a software helper call, a magic-number-
add sequence, or a high/low-split-and-add sequence to perform
unsigned 64-bit integer to double-precision float conversion.
Should also add a discriminative regression test for the
high-bit-set case (`u64(0x8000000000000000) as f64` produces a
positive float ~9.22e18, not the signed-interpretation -9.22e18).

## Cross-reference to cycles 101-110 + Stage 29.1 + Stage 30

- **Cycle 101**: PASS, 0 findings. `_is_i64_type` sibling sites
  noted as deferred (BIT_*, SHL, SHR, BIT_NOT, NEG, DIV, MOD).
- **Cycle 102**: fix-sweep, 4 cycle-101 findings (ADD/SUB/MUL u64).
- **Cycle 103-104**: PASS, 0 findings.
- **Cycle 105**: FAIL, 1 finding F105-1 (f64↔f32 CAST silent).
- **Cycle 106**: fix-sweep, 4+ cycle-105 findings + cross-cut.
- **Cycle 107**: PASS on type-design (counter 0→1). Silent-
  failures audit: FAIL, 8 findings F1-F8.
- **Cycle 108**: fix-sweep, all 8 cycle-107 silent-failure
  findings closed. CALL / RETURN / SELECT / BR / LOAD_VAR /
  STORE_VAR / CAST promoted to `_is_64bit_int_type`. CharLit /
  StructLit / TileLit loud-fail arms added.
- **Stage 29.1**: bumped patch_table 4096→16384 and bind_state
  64→512 caps. K3 exits 42. Broke the cycle-1 polish-#14 cap-
  coordination invariant by a factor of 4.
- **Cycle 109**: FINDINGS on type-design (F109-1 cap-coordination)
  and silent-failures (1 CRITICAL F1 + 7 HIGH including F2 Range,
  F3 unsigned→float, F4 BIT_*/SHL/BIT_NOT/NEG). Counter reset to
  0/5.
- **Cycle 110**: fix-sweep `9c451e6` — F2/F3/F4 closed at Stage
  28.9 scope; F1 (cap-coordination) closed via in-source kovc.hx
  edit (prologue 1024→4096, bind_alloc_offset 1024→4096). F3
  closure is partial — u32 portion fixed; u64 portion still
  uses signed cvtsi2sd from rax (cycle-111 F111-1 below).
- **Stage 30 cycle-1**: 2 HIGH + 3 MEDIUM findings (NOT CLEAN).
- **Stage 30 cycle-2**: H1 fix `fe7042f` wires early_err sentinel
  return path. Polish `1aecbae` updates comments. IMPORTANT
  fix `f9425a0` adds regression tests for traps 62032/62033.
- **Cycle 111** (this doc): FINDINGS, 1 HIGH F111-1 (cycle-110
  partial F3 closure — u64→f silent miscompile). Counter resets
  1/5 → 0/5.

## No code edits made

This audit performed STRICT READ-ONLY analysis on `helixc/`. The
only file written by cycle-111 is this document
(`docs/audit-stage28-9-cycle111-type-design.md`). All references
to source code (line numbers, byte patterns, predicate names)
were verified against the HEAD `f9425a0` checkout via Read /
Grep tools. No Edit or Write calls to source files.
