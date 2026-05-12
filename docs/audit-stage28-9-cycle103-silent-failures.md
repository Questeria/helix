# Audit Stage 28.9 cycle 103 — Silent failures

**Date:** 2026-05-12
**HEAD:** `26dfa82` (`Stage 28.9 cycle-102 fix-sweep: 4 cycle-101 findings (ADD/SUB/MUL u64 + regression tests)`)
**Counter at start:** 0/5 (cycle-101 FAIL → reset by cycle-102 fix-sweep)
**Bar:** ZERO new findings at confidence ≥ 75.

**Scope (narrow, per prompt):**
- `helixc/backend/x86_64.py` cycle-102 delta (31 LOC: new `_is_64bit_int_type` helper at line 1033 and ADD/SUB/MUL dispatch sites at lines 1329/1359/1387).
- `helixc/tests/test_ir.py` cycle-102 delta (50 LOC: `test_c100_unsigned_cmp_emits_setb_not_setl` and `test_c102_u64_add_emits_64bit_path`).

**Mode:** STRICT READ-ONLY. No source edits performed. Single Write of
this audit doc. Source files only read/grepped. Two unit-test probes
(predicate-reversion fault injection) executed under `python -c …` to
validate that the new regression tests are real fail-gates rather than
tautological assertions — no source mutation, no scorecard run.

---

## Methodology

1. **Read the cycle-102 diff in isolation** (`git show 26dfa82 -- helixc/backend/x86_64.py helixc/tests/test_ir.py`).
2. **Read the surrounding helper definitions** (`_is_i64_type` at 1019, `_is_u64_type` at 1027, `_is_64bit_int_type` at 1033, `_is_unsigned_int_type` at 1044) to confirm the new helper is a faithful union of two well-tested predicates.
3. **Read the ADD/SUB/MUL dispatch sites in full context** (lines 1315–1403) to confirm dispatch order (f64 → float → 64-bit-int → 32-bit-int else) is unchanged and that the new helper only widens membership of the 64-bit-int arm.
4. **Cross-reference cycle-101 silent-failures and codereview findings** to confirm cycle-102 closed the 3 in-scope items and deferred 2 (DIV/MOD signed-vs-unsigned, SHR signed-vs-unsigned, StrLit lowering gap) per its own commit message.
5. **Fault-injection probes** to confirm the two new regression tests would fail on the pre-fix predicate:
   - Monkey-patched `_is_64bit_int_type` back to `_is_i64_type` only → ELF emits `01 C8` (32-bit add), no `48 01 C8`; test would FAIL pre-fix.
   - Monkey-patched `_is_unsigned_int_type` to always return False → ELF emits `0F 9C C0` (setl), no `0F 92 C0` (setb); test would FAIL pre-fix.
6. **Ran `python helixc/tests/test_ir.py`** to confirm post-fix passes (17/17 passed).
7. **Looked for interaction effects** between the new ADD/SUB/MUL u64 path and the still-broken DIV/MOD/SHR paths (deferred per cycle-102 commit) — see §"Interaction effects" below.
8. **Manufactured-finding guard:** verified no candidate finding was a re-skin of a cycle 1–102 finding (especially cycle-101 F1, cycle-101 F2, cycle-101 codereview F1, cycle-101 codereview F2).

---

## Cycle-102 fix verification

| Finding | Status | Witness |
|---|---|---|
| cycle-101 silent-failures F1 (StrLit, HIGH 85) | DEFERRED (per cycle-102 commit) | Not in cycle-102 diff; explicitly listed as deferred in commit body. |
| cycle-101 silent-failures F2 (ADD/SUB/MUL u64 width, HIGH 92) | CLOSED | New `_is_64bit_int_type` helper at line 1033; ADD/SUB/MUL each switched from `_is_i64_type` to `_is_64bit_int_type` at lines 1329/1359/1387. Helper is `_is_i64_type(ty) or _is_u64_type(ty)` → `{i64, isize, u64, usize}`, matching cycle-101 F2's prescribed fix shape verbatim. |
| cycle-101 codereview F1 (cycle-100 missing regression test, HIGH 90) | CLOSED | Two new tests in `test_ir.py`. Probe confirms `test_c100_unsigned_cmp_emits_setb_not_setl` fails pre-cycle-100, passes post-cycle-100. |
| cycle-101 codereview F2 (DIV/MOD/SHR signed-only, HIGH 85) | DEFERRED (per cycle-102 commit) | Not in cycle-102 diff; explicitly listed as deferred in commit body. |

Mechanical correctness of the new helper:
- `_is_i64_type` → `{i64, isize}` (cycle-19 C18-1 fix, well-audited).
- `_is_u64_type` → `{u64, usize}` (cycle-19 C18-1 fix, well-audited).
- `_is_64bit_int_type` = union → `{i64, isize, u64, usize}`. Type-restricted to `TIRScalar` because both inputs are.

The cycle-102 dispatch widening is sign-agnostic at the machine level for ADD/SUB/MUL low-half results, which is correct for these three opcodes:
- `add r/m64, r64` (REX.W 01 /r) and `sub r/m64, r64` (REX.W 29 /r): the same opcode is used for signed and unsigned addition / subtraction; the result low 64 bits are bit-identical for both interpretations (2's complement wraparound).
- `imul r64, r/m64` (REX.W 0F AF /r): the low 64 bits of an N×N→2N multiplication agree between signed and unsigned interpretations under 2's complement. Backend captures only the low 64 bits into the result slot; the high half (which would require `mul` for unsigned) is discarded.

So the fix is correct in shape for all three.

---

## Regression-test fitness

Both new tests probe ELF bytes for opcode sequences. Fault-injection probes (monkey-patching the relevant predicate to pre-fix behavior) show:

- `test_c102_u64_add_emits_64bit_path`: post-fix ELF contains `48 01 C8`; pre-fix ELF contains `01 C8` and lacks `48 01 C8`. The byte pattern `48 01 C8` is `REX.W ADD rax, rcx` — a specific enough opcode triple that incidental collisions in a tiny two-function ELF are improbable (no `48` REX.W prefix appears adjacent to a `01` ADD opcode anywhere else in the entry stub or `main` body for this source).
- `test_c100_unsigned_cmp_emits_setb_not_setl`: post-fix ELF contains `0F 92 C0` (setb al); pre-fix ELF contains `0F 9C C0` (setl al) without `0F 92 C0`. `0F 92` is a 2-byte opcode escape; collision risk negligible.

Both tests are real fail-gates, not no-op assertions.

---

## Interaction effects with deferred DIV/MOD/SHR paths

The cycle-102 fix did NOT modify DIV/MOD/SHR. Lines 1404–1444 (DIV/MOD) and 1509–1523 (SHR) still gate on `_is_i64_type` only and still emit signed `idiv` / `sar` unconditionally — both deferred per the cycle-102 commit and tracked as cycle-101 codereview F2. No re-flag.

Question: does cycle-102's ADD/SUB/MUL widening expose any latent bug in the deferred ops?

- The new helper is a pure read-only predicate; it adds no shared mutable state.
- ADD/SUB/MUL and DIV/MOD/SHR are separate `OpKind` arms with independent slot-of / register-use sequences. No `rax`/`rcx`/`rdx` aliasing risk introduced.
- The widening cannot affect type propagation: `lower_ast.py:1119` already emitted `result_ty=l.ty` for these ops, so the IR shape feeding the backend is identical pre- and post-cycle-102.
- No call path from ADD/SUB/MUL into the DIV/MOD/SHR emitters exists.

Conclusion: zero interaction. The deferred paths' miscompiles are unchanged in surface and reachability — they remain bugs of identical severity to what cycle-101 codereview F2 documented, and remain off-scope for cycle-103.

---

## Other things considered and ruled out

- **`_is_64bit_int_type` introducing a wrong-arm dispatch via `TIRScalar` check on a non-scalar type.** Both `_is_i64_type` and `_is_u64_type` already gate on `isinstance(ty, tir.TIRScalar)`, so the union returns False for `TIRTensor` / `TIRTile` / `TIRRef` / `TIRUnit`. Aggregate types fall through to the existing 32-bit else branch, which itself rejects non-scalar at the slot-mov level. No new wrong-arm dispatch.
- **`imul r64, r/m64` high-half discrepancy on unsigned.** For a u64×u64 source that overflows 64 bits, the user-visible result is the low 64 bits regardless of signed vs unsigned, because the helixc IR has no `MUL_WIDE` / two-result MUL op that could expose the high half. So `imul` and `mul` agree on the captured result. (`SUB` and `ADD` are bit-identical between signed and unsigned at all widths.) Not a finding.
- **Test coverage gap: SUB and MUL u64 paths have no regression test.** Cycle-102 added a u64 ADD regression but no u64 SUB or u64 MUL regression. A reviewer reverting only the SUB or MUL `_is_64bit_int_type` call back to `_is_i64_type` would not be caught by the suite. However: (a) the three sites are line-for-line identical and shipped together in one commit, (b) the test_c102 docstring explicitly names "ADD/SUB/MUL" as the fix scope, signalling the SUB/MUL coverage is intentionally implied by the ADD probe, (c) cycle-101 codereview F1's bar was "at least one regression test that fails pre-fix and passes post-fix" — that bar is met. Confidence this is a HIGH-class silent-failure finding: ~55 (below the 75 bar). Not flagged.
- **`48 01 C8` byte-pattern false-positive in the test.** A two-function ELF (`add_u64` + `main`) is ~4.7 KB; the literal byte triple `48 01 C8` could in principle appear in PT_LOAD padding, .shstrtab, or the entry stub. Inspection of the entry stub (which uses `mov edi, eax` + `mov eax, 60` + `syscall`) shows no `48 01 C8` adjacent triple; the `main` body returns 0 with no ADD; only `add_u64`'s body produces this opcode. False-positive risk is low. Confidence this is a HIGH-class finding: ~35. Not flagged.
- **The new `_is_64bit_int_type` helper does NOT replace `_is_i64_type` in the still-i64-only DIV/MOD/SHR/BIT_AND/BIT_OR/BIT_XOR/SHL/BIT_NOT/NEG arms.** DIV/MOD/SHR are explicitly deferred (cycle-101 codereview F2). BIT_AND/BIT_OR/BIT_XOR/SHL/BIT_NOT/NEG also still gate on `_is_i64_type` only — but these are NOT part of the cycle-102 delta and have never been flagged in any prior cycle audit. The cycle-103 scope is strictly the cycle-102 delta; expanding scope to these untouched ops would be scope creep and is ruled out per the prompt's "Focus on the cycle-102 delta" instruction. (Per the "manufacturing findings is forbidden" guard: the BIT_*/SHL/NEG/BIT_NOT pattern is the same defect *class* as cycle-101 F2, but it is not part of the cycle-102 delta and re-discovering it here would not be a cycle-102-introduced silent failure. If a future cycle's rotation lands on these ops in the prompt's scope, that is the appropriate place to surface them.)

---

## Findings table

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH (conf ≥ 75) | 0 |
| MEDIUM (60–74) | 0 surfaced (one near-miss on SUB/MUL test coverage at ~55) |
| LOW (<60) | 0 surfaced |

---

## Verdict

**CLEAN** — 0 findings at confidence ≥ 75 within the cycle-102 delta.

Counter advances: 0/5 → 1/5.

---

## Cross-references

- Cycle-101 silent-failures audit: `docs/audit-stage28-9-cycle101-silent-failures.md` (FAIL, 2 findings; one closed in cycle-102, one deferred).
- Cycle-101 codereview audit: `docs/audit-stage28-9-cycle101-codereview.md` (FAIL, 2 findings; one closed in cycle-102, one deferred).
- Cycle-102 commit: `26dfa82` (`Stage 28.9 cycle-102 fix-sweep: 4 cycle-101 findings (ADD/SUB/MUL u64 + regression tests)`).
- Predicate provenance: cycle-19 C18-1 (introduced `_is_i64_type`/`_is_u64_type` with `isize`/`usize` aliasing). Cycle-100 (`caf203f`) introduced `_is_unsigned_int_type` for cmp dispatch. Cycle-102 (`26dfa82`) introduced `_is_64bit_int_type` for ADD/SUB/MUL.
