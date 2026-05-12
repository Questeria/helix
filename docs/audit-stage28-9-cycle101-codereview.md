# Audit Stage 28.9 cycle 101 — Code review

Scope: HEAD `fbfa211`. Cycle-100 commit under review: `caf203f` (Stage 28.9
cycle-100 fix-sweep — F1/F2 unsigned int cmp). Strict read-only audit.
ONE Write of this doc; no Edit, no source mutation, no test run.

Prior C1–C100 findings + deferred-known carve-outs NOT re-flagged.
Independent Stage 28.10 / 28.11 / 28.13.1 cycle activity ignored.

Dimensions reviewed (NARROW):
1. Clarity of comments on `_is_unsigned_int_type` predicate and
   `unsigned_int_cmp_setters` dispatch table added in cycle-100.
2. Presence of regression tests for cycle-100 in
   `helixc/tests/test_codegen.py`.
3. Other backend pairs in `helixc/backend/x86_64.py` exhibiting the same
   signed-only handling pattern that the cycle-100 split should mirror.

---

## Verdict: FAIL — 2 findings at conf >= 75%

### F1 (HIGH, conf 90%) — cycle-100 ships no regression test in `test_codegen.py`

File: `helixc/tests/test_codegen.py` (no diff). The cycle-100 commit
(`caf203f`) modifies `helixc/backend/x86_64.py` only; the file-stat
listing shows `helixc/backend/x86_64.py` plus three `docs/audit-stage28-9-
cycle99-*.md` files and zero test edits. A grep for `_is_unsigned_int_type
` / `unsigned_int_cmp_setters` / `0xFFFFFFFF_u32 < 1_u32` / `cycle 100` /
`cycle-100` returns no hits in `test_codegen.py`. Pre-existing u32/u64
cmp tests at lines 2327–2334 and 3171–3182 use small values (5, 10) where
signed and unsigned setcc happen to agree — cycle-99's F1/F2 write-up
explicitly calls out lines 3172–3182 as a "test passes despite the bug"
witness. So the cycle-100 fix has zero direct executable witness in the
suite; the heavy-gate "1522 passed" claim in the commit message covers
the regression only by not regressing pre-existing passing tests, not by
asserting the buggy behavior would have been rejected. A high-bit-set
unsigned-cmp test (e.g. `let a: u32 = 0xFFFFFFFF_u32; if a < 1_u32 { 0 }
else { 42 } == 42`, plus the u64 analog `0x1_0000_0000_u64 < 1_u64`) is
trivial to add and would have failed pre-`caf203f` and now passes — the
exact regression test the cycle-19 C18-1 and cycle-93 fixes both included
as a matter of convention (see `tests/test_codegen.py:478` for the C18-1
precedent: a dedicated `_is_i64_type` / `_is_u64_type` predicate probe
test). Cycle 100 broke that convention.

### F2 (HIGH, conf 85%) — DIV / MOD / SHR in `helixc/backend/x86_64.py` have the same signed-only pattern cycle-100 just fixed for CMP

File: `helixc/backend/x86_64.py` lines 1380–1418 (DIV / MOD) and 1484–
1497 (SHR). The cycle-100 fix split the cmp dispatch on
`_is_unsigned_int_type` AND extended the 64-bit gate from
`_is_i64_type(...)` to `_is_i64_type(...) or _is_u64_type(...)`. The
same two defects survive verbatim in the sibling arithmetic / shift
arms:

- **DIV** (line 1393): `elif self._is_i64_type(op.results[0].ty): ...
  cqo + idiv rcx` then else-fall-through to `_emit_idiv_guarded` which
  emits `cdq + idiv ecx`. (a) `u64` / `usize` operands fall to the
  32-bit `idiv` path (silent high-half truncation, same class as
  cycle-100 F1). (b) `u32` / `u64` operands use **signed** `idiv`
  unconditionally; the unsigned-division opcode is `div` (REX.W or
  32-bit); for high-bit-set u32 / u64 dividends `idiv` reads the value
  as negative and produces wrong quotient + sign (same class as
  cycle-100 F2). Cycle-99's bootstrap-side u32 stage comment at
  `test_codegen.py:2316` already documents this same dispatch
  requirement on the bootstrap (`xor edx, edx; div ecx`), proving the
  semantic expectation, but the Python backend does not honor it.

- **MOD** (line 1408): identical pattern — `cqo + idiv rcx` for i64,
  `_emit_idiv_guarded` (signed) for everything else. Same two defects.

- **SHR** (line 1484): `if self._is_i64_type(...) ... sar_rax_cl()
  else: ... sar_eax_cl()`. `sar` is arithmetic (signed) shift right.
  For `u8` / `u16` / `u32` / `u64` / `usize` the correct opcode is
  `shr` (logical). High-bit-set unsigned values right-shift with the
  high bit replicated, miscompiling the result (e.g. `0xFFFFFFFF_u32
  >> 1` should be `0x7FFFFFFF`, but `sar` returns `0xFFFFFFFF`). Plus
  the 64-bit gate misses `u64` / `usize` exactly as cmp did.

These three ops are the structurally identical siblings of the CMP arm
cycle-100 just patched. Cycle-100 introduced the helper
(`_is_unsigned_int_type`) and the conceptual fix; failing to apply it
to DIV / MOD / SHR in the same sweep leaves three more known-shape
silent miscompiles live for arbitrary user code. Grep `idiv\|sar` of
`helixc/backend/x86_64.py` confirms no `div` / `shr` (unsigned) opcode
mnemonics exist in `Asm` — the backend has **no** unsigned division /
logical shift emission today. This is genuinely new (not deferred-
known); cycle-99 silent-failures audit scope explicitly narrowed to
"integer comparison emit (CMP_LT/GT/EQ)" only, leaving DIV / MOD / SHR
unaudited. No prior cycle audit doc grep (`audit-stage28-*`) flags
unsigned DIV / SHR in the Python backend.

---

## Positive observations (no finding)

- Comments on the new `_is_unsigned_int_type` (`x86_64.py:1033-1045`)
  and `unsigned_int_cmp_setters` (`x86_64.py:1568-1587`) and on the
  patched dispatch site (`x86_64.py:1636-1647`) reference the cycle,
  finding ID, confidence band, and a concrete miscompile example
  (`0xFFFFFFFF_u32 < 1_u32`). Comment quality is in line with the
  surrounding cycle-19 / cycle-93 / cycle-97 audit-comment style.
- The cmp 64-bit dispatch correctly uses **either operand** to widen,
  matching the cycle-19 C18-1 fix shape.

---

## No code edits made

This audit is strict read-only. ONE Write of this doc only — no Edit
calls, no source mutation, no test run, no scorecard run. Surfaces F1
(missing regression test) and F2 (DIV / MOD / SHR signed-only pattern)
to the Stage 28.9 fix-sweep for the next cycle.
