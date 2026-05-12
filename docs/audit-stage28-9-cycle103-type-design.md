# Audit Stage 28.9 cycle 103 — Type design

## Header

- **Date**: 2026-05-12
- **HEAD**: `26dfa82` ("Stage 28.9 cycle-102 fix-sweep: 4 cycle-101
  findings (ADD/SUB/MUL u64 + regression tests)")
- **Counter at start**: 0/5 (cycle-101 FAIL → reset by cycle-102
  fix-sweep)
- **Scope**: narrow type-design audit of the cycle-102 delta:
  the new `_is_64bit_int_type` helper in
  `helixc/backend/x86_64.py`, the ADD/SUB/MUL emit-site switch
  from `_is_i64_type` to `_is_64bit_int_type`, and the two new
  ELF-byte regression tests in `helixc/tests/test_ir.py`.
- **Bar**: PASS = ZERO new findings at confidence ≥ 75%.
  Re-flagging cycle 1-102 findings is FORBIDDEN.
- **Mode**: read-only on `helixc/`. Only write is this document.
- **Explicitly out of scope per prompt**: A.StrLit lowering gap,
  DIV/MOD/SHR signed-vs-unsigned (deferred-known from cycle-57
  and cycle-101).

## Methodology

Four verification points covering the cycle-102 surface:

1. **V1 — `_is_64bit_int_type` semantics**: enumerate every
   64-bit integer scalar type Helix can emit at the
   `TIRScalar.name` layer, and check the helper's `("i64",
   "isize", "u64", "usize")` set against that enumeration.
2. **V2 — `_is_i64_type` cross-check**: locate every remaining
   `_is_i64_type` call site post-cycle-102 and classify each as
   either (a) appropriate (i64-specific, e.g. cast-matrix dispatch
   keyed on signedness), or (b) latent-defect already deferred
   from cycle-57/cycle-101, or (c) net-new defect uncovered by
   cycle-102 widening (would-be flagged).
3. **V3 — Sign-correctness of `add`/`sub`/`imul`-low**: verify
   the commit message's claim that the three opcodes the
   cycle-102 fix routes through are sign-agnostic at the low-N-bit
   width, against the x86-64 architecture reference.
4. **V4 — Test invariants**: judge whether the two new ELF-byte
   inspection tests check meaningful invariants or are
   tautological re-writes of the codegen emission.

## Findings table

| ID | Severity | Confidence | Topic | Disposition |
|----|----------|------------|-------|-------------|
| —  | —        | —          | —     | No findings at conf ≥ 75 |

## Verification points (detailed)

### V1 — `_is_64bit_int_type` type-set completeness (PASS)

`helixc/backend/x86_64.py:1033-1042` defines the helper as
`isinstance(ty, tir.TIRScalar) and ty.name in {"i64", "isize",
"u64", "usize"}` (via composition of `_is_i64_type` and
`_is_u64_type`). The full set of integer-shaped primitives
Helix can produce at the `TIRScalar.name` layer is established
by three independent sources, all in agreement:

- `helixc/ir/lower_ast.py:356-362` — `_PRIMITIVE_TYPE_NAMES`
  frozenset: `{i8, i16, i32, i64, isize, u8, u16, u32, u64,
  usize, bool, char, bf16, f16, f32, f64, unit}`.
- `helixc/frontend/typecheck.py:336-343` — `PRIMITIVES`:
  identical integer half `{i8…isize, u8…usize}` plus
  `{bool, char}` and floats `{bf16, f16, f32, f64, fp8, mxfp4,
  nvfp4, ternary, ()}`.
- `helixc/frontend/typecheck.py:2138-2141` —
  `_NUMERIC_INT_PRIMS`: `{i8, i16, i32, i64, isize, u8, u16,
  u32, u64, usize}`.

The 64-bit integer subset of `_NUMERIC_INT_PRIMS` is exactly
`{i64, isize, u64, usize}` — `isize`/`usize` are
pointer-width-aliased to i64/u64 on 64-bit targets (cycle-19
C18-1 disposition, comment in `_is_i64_type`/`_is_u64_type`).
Helix has no `c_long`, `u128`, `ssize`, or other 64-bit-shaped
primitive at the `TIRScalar` layer. Raw pointers
(`A.TyPtr`) lower to `TIRScalar("u64")` per
`lower_ast.py:391-394` — already covered.

Generic type parameters (`fn id[T](x: T)`) lower to
`TIRScalar(ty.name)` where `ty.name` is the param identifier
(e.g. `"T"`); these will NOT match the helper even if the type
arg is u64 at call-site. This is the **monomorphisation-gap
HBS limitation** explicitly documented in `lower_ast.py:351-355`
("silently lower to TIRScalar('T') with i32-sized ABI today; this
is correct for i32 type args and silently wrong for i64+"). It
predates cycle-102 and is not a defect of the helper.

**Type set closed against Helix's frontend.** PASS at conf ≥ 75.

### V2 — `_is_i64_type` cross-check post-cycle-102 (PASS)

Post-cycle-102 `_is_i64_type` is referenced at 19 sites in
`x86_64.py` (per grep). Classification:

**Cycle-102 switched sites (3, all to `_is_64bit_int_type`)**:
ADD@1329, SUB@1359, MUL@1387. Verified in
`git show HEAD -- helixc/backend/x86_64.py` — each elif was
replaced with the new helper.

**Sites that remain `_is_i64_type` correctly (signedness-specific
or cast-matrix keyed on signed half)**:
- 986 — function-prologue parameter spill width (i64/isize get
  the 8-byte slot move; u64/usize handled separately at the FFI
  layer per `_is_u64_type` and at the calling-convention layer).
  Note: this is a latent cycle-57-deferred site for the bare-IR
  path (`let p: u64 = ...; fn(p)`), tracked under the same
  deferred-known sweep. NOT cycle-102 territory.
- 1253-1254 (`from_is_i64`, `to_is_i64`) — CAST i64↔i32 / i64→f64
  / i64↔i64 cases. These are signed-cast paths
  (`movsxd` zero-extends from the signed-low half via sign bit;
  `cvtsi2sd` is signed-int-to-fp). Routing u64→f64 through
  `cvtsi2sd` is semantically wrong for u64 values ≥ 2^63, which
  is a known cycle-57-deferred cast-matrix gap. NOT cycle-102
  territory.

**Sites with same `_is_i64_type`-only fallthrough pattern, all
explicitly deferred-known from cycle-57 + cycle-101 V3 list**:
1418 (DIV), 1433 (MOD), 1453 (BIT_AND), 1468 (BIT_OR), 1483
(BIT_XOR), 1498 (SHL), 1513 (SHR), 1527 (BIT_NOT), 1540 (NEG),
1198 (CONST_INT), 1234-1236 (BITCAST `wide`), 1716 (RETURN-path
res_ty wide gate), 1816 (CALL-path res-store wide gate). All
appear in cycle-57's "<75 Notes" section (lines 144-165 of
`docs/audit-stage28-9-cycle57-type-design.md`) and the
cycle-101 V3 "deferred-known" list (lines 39-49 of
`docs/audit-stage28-9-cycle101-type-design.md`).
Cycle-102's commit-message deferred-section explicitly names
DIV/MOD/SHR. Per cycle-103 scope, re-flagging is FORBIDDEN.

**Sites widened inline (without the helper)**: 1672-1676,
1872, 1891 — cmp dispatch (cycle-100 fix) and CALL-arg
register-load (Stage 16.5 FFI) already use the
`_is_i64_type(...) or _is_u64_type(...)` pattern directly.
Cycle-102 did not promote these to the helper; they remain
correct but stylistically inconsistent with the new helper.
This is a **minor consistency drift** (the helper now exists,
so the inline OR could be folded into it), but it is not a
type-design defect — both forms produce identical truth tables.
Below 75 as a finding (style-only, no observable behavior
change).

**No net-new defect uncovered by cycle-102 widening.** The
switch correctly broadens ADD/SUB/MUL to the four-element set
without introducing any new mis-classification. PASS at conf ≥ 75.

### V3 — Sign-correctness of `add`/`sub`/`imul`-low (PASS)

Cycle-102 routes u64/usize through the existing 64-bit
opcodes `add rax, rcx` (`48 01 C8`), `sub rax, rcx`
(`48 29 C8`), and `imul rax, rcx` (`48 0F AF C1`). The
commit message claims these are sign-agnostic at the
machine level.

- **`add` / `sub`**: Intel SDM Vol. 2 documents a single opcode
  per width for add and sub; the low-N-bit result is identical
  for signed and unsigned operands (the CF flag captures
  unsigned overflow, OF captures signed overflow, but the
  destination bits are the same). This is a property of two's
  complement: the bit-level addition and subtraction circuits
  produce identical low-half results regardless of how the
  operands are interpreted. ✓
- **`imul` low-half == `mul` low-half**: For the 2-operand
  `imul r64, r64` form, the destination receives the low 64
  bits of the 128-bit product. Two's complement multiplication
  theorem: the low N bits of signed and unsigned product on
  N-bit operands are bit-identical (only the high N bits and
  sign-extension behavior differ). Since the cycle-102 path
  stores only `rax` (the low-half), and never reads `rdx` (the
  high-half), the same opcode is correct for u64 and i64. ✓

The cycle-102 comment at the MUL site (line 1389-1393) explicitly
notes this: "imul lower-half is identical for signed and unsigned
operands (only upper-half via mul vs imul differs, which we
don't capture in single-result use)". Comment is accurate.

No overflow-flag observation is performed (no `jo`/`jno` follows
the arithmetic), so the OF/CF distinction is unreachable in
emitted code. Helix has no `checked_add` / `wrapping_add`
distinction at Phase-0; all arithmetic is wrapping.

**Claim verified.** PASS at conf ≥ 75.

### V4 — Test invariants (PASS)

Two new tests in `helixc/tests/test_ir.py`:

**`test_c100_unsigned_cmp_emits_setb_not_setl`** (lines 232-253):
asserts the ELF contains `\x0f\x92\xc0` (setb al). Pre-fix the
cycle-100 codegen emitted `\x0f\x9c\xc0` (setl al) for u32
operands; post-fix it emits `\x0f\x92\xc0`. The two byte
sequences differ only at the second byte (`9c` → `92`), so the
test discriminates between the pre-fix and post-fix output. The
test would FAIL on the pre-fix codegen (the `setb` opcode is
absent — only `setl` would appear in the u32-cmp body) and
PASS on the post-fix codegen. **Discriminating, not
tautological.**

**`test_c102_u64_add_emits_64bit_path`** (lines 255-282):
asserts the ELF contains `\x48\x01\xc8` (rex.W add rax, rcx).
Pre-fix the cycle-102 codegen would have emitted `\x01\xc8`
(`add eax, ecx`, no rex.W) for u64 ADD, falling through to the
32-bit path. The presence of the `\x48` prefix byte immediately
before `\x01\xc8` is the structural difference between the
32-bit and 64-bit emit paths. The test would FAIL on the
pre-fix codegen (no `0x48` rex.W prefix on the u64-add bytes)
and PASS on the post-fix codegen. **Discriminating, not
tautological.**

Minor caveat (below 75): both tests use `in elf` substring
match against the whole ELF binary, so a coincidental
appearance of the three target bytes in rodata or in an
unrelated function's body would mask a regression. For the
specific 3-byte sequences `0F 92 C0` and `48 01 C8` in a short
single-function ELF, the false-positive rate is negligible —
and the cycle-19/93/100 regression-test convention has
established the same opcode-substring pattern (this is the
codebase's idiom, not a cycle-102 invention). Test-quality
observation, not a type-design defect.

**Both tests check meaningful, fix-specific invariants.** PASS
at conf ≥ 75.

## Verdict

**PASS** — 0 new findings at confidence ≥ 75.

- V1 confirms the `_is_64bit_int_type` type-set
  `{i64, isize, u64, usize}` is closed against Helix's
  frontend primitive enumeration.
- V2 confirms cycle-102 widened ADD/SUB/MUL without introducing
  any net-new mis-classification; remaining `_is_i64_type`-only
  sites are either correctly signedness-specific or
  deferred-known from cycle-57/cycle-101.
- V3 confirms the sign-agnostic claim for `add`/`sub`/`imul`-low
  against Intel SDM and two's-complement arithmetic.
- V4 confirms the two new ELF-byte tests are discriminating,
  not tautological — each would fail on the pre-fix codegen.

Stage 28.9 audit-gate counter advances **0 → 1**.

## Cross-reference to cycle 101 / 102

- **Cycle 101 type-design** (`fbfa211`,
  `docs/audit-stage28-9-cycle101-type-design.md`): PASS, 0
  findings. V3 enumerated the wider `_is_i64_type`-only
  fallthrough sites (DIV/MOD/BIT_*/SHL/SHR/BIT_NOT/NEG) as
  deferred-known from cycle-57.
- **Cycle 101 silent-failures**
  (`docs/audit-stage28-9-cycle101-silent-failures.md`): FAIL,
  2 findings — F1 (A.StrLit lowering gap) and F2 (ADD/SUB/MUL
  u64/usize fallthrough). Cycle-102 fix-sweep remediated F2
  via the new helper.
- **Cycle 102** (`26dfa82`, this HEAD): introduces
  `_is_64bit_int_type` helper, switches ADD/SUB/MUL, adds two
  ELF-byte regression tests. Heavy gate post-fix: 1523 passed.
- **Cycle 57 type-design**
  (`docs/audit-stage28-9-cycle57-type-design.md`, "<75 Notes",
  lines 144-165): the foundational documentation of the
  `_is_i64_type`-only-vs-u64 deferred-known defect class.
  Cycle-102's helper is the cycle-58-recommended remediation
  shape (`_is_64bit_int_type` predicate), now realized for
  ADD/SUB/MUL and to be propagated in future cycles for the
  remaining sites.

No prior-cycle findings re-surface; no edits to source
performed; this document is the only file written.
