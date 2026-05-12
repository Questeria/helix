# Audit Stage 28.9 cycle 104 — Code review

- **Date**: 2026-05-12
- **HEAD**: `31e1725` ("Stage 28.9 cycle-103 audits: 3/3 CLEAN, counter 0/5 → 1/5")
- **Counter at start**: 1/5 (cycle-103 CLEAN)
- **Scope**: `git diff 26dfa82..HEAD -- helixc/`
  - Empty. No `helixc/` source changes between cycle-103 baseline (`26dfa82`) and cycle-104 HEAD (`31e1725`). Cycles 103 and `efc9be6` (28.11 cycle-5 audit-doc straggler) shipped pure markdown deltas in `docs/`.
  - Per cycle-104 prompt the audit surface is widened to recent feature work: Stage 28.9 (u64 arithmetic — `26dfa82`), Stage 28.10 (PAT_OR — `40289d6` and predecessors), Stage 28.11 (generic structs — `7123f09` and predecessors), Stage 28.13.1 / 28.13.2 (named struct-lit — `30c4bc0`, `fbfa211`, `4b938d2`).
  - Cross-stage cuts: (a) generic-mono ↔ backend type predicates (`_is_64bit_int_type` + the deferred `_is_i64_type`-only sibling family); (b) PAT_OR drain interactions with `bn_state` threading.
- **Bar**: only report findings at confidence ≥ 80. Re-flagging C1–C103 findings is forbidden. Specifically excluded from re-flag: cycle-101 codereview F2 tail (DIV / MOD / SHR / BIT_AND / BIT_OR / BIT_XOR / SHL / BIT_NOT / NEG signed-vs-unsigned / u64-width gate), cycle-101 silent-failures F1 (A.StrLit IR lowering gap), cycle-57 deferred-known `_is_i64_type`-only fallthrough sites (CONST_INT@1198, BITCAST@1234-1236, RETURN@1716, CALL@1816, prologue spill@986, cast matrix `from_is_i64`/`to_is_i64` @1253-1254), cycle-103 sub-threshold inline-`or _is_u64_type` style drift at 1672-1676 / 1872 / 1891.

---

## Methodology

Read-only inspection. No source mutation, no test runs, no scorecard. One Write for this doc.

Because no `helixc/` source moved between cycle-103 and cycle-104, the audit re-walks the cycle-102 delta (the most recent landed source change) under freshly chosen dimensions that cycle-103's codereview pass did NOT explicitly probe, then widens to the cross-stage cuts in the prompt:

1. **Predicate-family invariant**: does `_is_64bit_int_type` participate in any iteration / closure / table construction where the union of two sub-predicates would behave differently than two separate membership tests?
2. **Bool / char width gate**: confirm `bool` and `char` (which are 8-bit per `typecheck.PRIMITIVES`) cannot accidentally satisfy any of the four 64-bit predicates and would-truncate-correctly through the 32-bit fallback.
3. **Cycle-102 helper coverage of `tir.TIRRef`/`TIRTensor`/`TIRTile`/`TIRUnit`**: the new helper unions two `isinstance(..., tir.TIRScalar)`-gated predicates. Confirm no path passes a non-scalar TIR type to ADD/SUB/MUL such that the cycle-102 widening could be exposed as a regression for a non-scalar arm.
4. **Generic-mono ↔ backend predicate**: Stage 28.11 INC-3b.2 (`7123f09`) introduced use-site `Pt<i32>` mono in `lower_ast.py` → produces `TIRScalar(name=...)` for the type-arg-instantiated field types. Confirm a `struct Pt<T> { x: T, y: T }` instantiated at `T=u64` flows through to ADD/SUB/MUL with the correct `TIRScalar("u64")` carrier so cycle-102's widening applies, and is not silently emitted as `TIRScalar("T")` per the documented monomorphisation-gap HBS limitation (`lower_ast.py:351-355`).
5. **Stage 28.13.2 named struct-lit ↔ field-order**: does the named-mode field-reorder in `parser.hx` parse_primary nt==16 produce a TUPLE_CONS whose result type carries the mono'd field types (not the generic-param identifiers)?
6. **Stage 28.10 PAT_OR drain ↔ `bn_state`**: confirm the `bn_state`-threading helpers added in cycle-90 (`ccfbd85`) consistently drain `or_alt` accumulator state across subpat helpers, especially on the first-position violation path (cycle-82 `82d673c`).
7. **Test placement consistency**: cycle-101 codereview F1 mandated the regression-test convention be `test_codegen.py`. Cycle-102 placed both new tests in `test_ir.py`. Cycle-103 disposed this below 75. Re-examine whether the placement could MASK a regression (e.g. by not exercising the subprocess-exit-code path the same way `test_codegen.py` does).
8. **`compile_module_to_elf` import locality**: each new test does a local `from helixc.backend.x86_64 import compile_module_to_elf`. Confirm this is not a leaky-test pattern that hides an import-time error.

---

## Findings table

| Severity   | Count |
|------------|-------|
| CRITICAL (90–100) | 0 |
| HIGH (80–89)      | 0 |
| MED (50–79)       | 0 reportable (below bar) |
| LOW (<50)         | 0 reportable (below bar) |

**Sub-threshold observations** (NOT findings — recorded for transparency only):

- **Cycle-104 has no source delta** (conf ~95 that this is *not* a finding): cycles 103 → 104 are pure markdown. Per the audit-gate counter rules, this means the 1/5 advance from cycle-103 is the only baseline change. There is no commit-introduced bug for cycle-104 to surface; the audit's job is to re-walk fresh dimensions and confirm no carry-over defect has been missed.
- **Stage 28.13.2 duplicated body parser** (conf ~55, explicitly declared in commit `4b938d2` body): the named-mode parsing logic in `parser.hx` parse_primary now exists in TWO places (non-generic nt==15-ish + generic-mono nt==16), ~80 LOC × 2 ≈ 160 LOC. Commit message acknowledges and defers to a future refactor extraction. Style / maintainability, not a defect; below 80.
- **Stage 28.11 INC-3b cap-overflow guard** (conf ~70 already-handled): cycle-3 (`549a68e`) closed five silent-failure findings; cycle-2 (`1ff41ff`) added the cap-overflow guard at the mono use site. No new gap surfaces in the cycle-104 read.
- **`bool` / `char` width-gate**: V2 confirms `bool` and `char` are not in any of the four 64-bit predicates' name sets (`_is_i64_type` → `{i64, isize}`, `_is_u64_type` → `{u64, usize}`, `_is_64bit_int_type` → union, `_is_unsigned_int_type` → `{u8, u16, u32, u64, usize}`). `bool` / `char` fall through to the 32-bit-int else branch, which is correct: both are 8-bit but the slot is 4-byte (mov eax / mov ecx); the upper-3-byte zero/garbage is masked by setcc / cmp behavior at the cmp-only consumer sites. No accidental wide-arithmetic dispatch.
- **Inline `or _is_u64_type` style drift at 1672-1676 / 1872 / 1891**: cycle-103 V2 already disposed this below 75 (style-only, identical truth table). Cycle-104 re-confirms — no new behavioral concern.
- **Test placement (test_ir.py vs test_codegen.py)**: cycle-103 disposed at conf ~55. Cycle-104 re-examination: the two cycle-102 tests assert on ELF bytes via `compile_module_to_elf(lower_src(src))`, which does NOT execute the ELF as a subprocess. `test_codegen.py` tests do subprocess-execute under Linux. Since the cycle-102 fix is a backend-emission predicate change with no runtime-only manifestation (the bug shape is "wrong opcode emitted" → byte-pattern is a sufficient witness), the test_ir.py placement is genuinely the right home: byte-inspection on Windows is portable; subprocess-exit-code on Windows is unrunnable. The cycle-101 codereview F1 bar (regression test exists, fails pre-fix, passes post-fix) is met. Below 80.
- **`compile_module_to_elf` local import**: cycle-103 disposed at conf ~40. Cycle-104 re-confirms — local-import idiom is harmless, no shared global state between tests.
- **Stage 28.10 PAT_OR `bn_state` threading**: cycle-90 (`ccfbd85`) closed the CRITICAL arity bug by threading `bn_state` through subpat helpers; cycles 91/93 verified the fix. Cycle-104 grep confirms the helper signatures consistently take and return `bn_state`; the cycle-82 drain-on-first-position-violation path (`82d673c`) routes through the same threaded state. No new gap.

---

## Cross-stage cuts examined

### Generic-mono ↔ backend type predicates (PASS at conf ≥ 80)

Stage 28.11 INC-3b.2 (`7123f09`) lowers `Pt<i32> { 10, 20 }` and Stage 28.13.2 (`4b938d2`) lowers `Pt<i32> { x: 10, y: 20 }` through `struct_mono.py` → field-type substitution → `TIRScalar("i32")` per mono'd field. The cycle-102 helper `_is_64bit_int_type` is keyed on `TIRScalar.name in {"i64", "isize", "u64", "usize"}`, so a mono at `T=u64` produces `TIRScalar("u64")` on each field — feed-forward into ADD/SUB/MUL the cycle-102 widening applies correctly.

The documented HBS limitation (`lower_ast.py:351-355`) — that a *non-monomorphised* generic param `T` lowers to `TIRScalar("T")` with i32 ABI — predates Stage 28.11 and is the pre-mono path. Post-mono (Stage 28.11 INC-3b.2 onward) the field types are substituted before reaching `TIRScalar`, so cycle-102's predicate matches the substituted name, not the type-param identifier. The integration is correct as designed.

### PAT_OR drain ↔ `bn_state` threading (PASS at conf ≥ 80)

Stage 28.10's cycle-82/cycle-85/cycle-90 fix-sweep established `bn_state`-threading as the canonical pattern for accumulator-drain on first-position violations. Cycle-91 audit (`3116b3a`, 4/5 clean) verified the arity fix; cycle-93 (`93f08dd`) confirmed stale-comment fix. No source change since Stage 28.10 closed (5/5 at `40289d6`). The PAT_OR surface is settled.

### Cycle-102 helper coverage for non-scalar TIR types

`_is_64bit_int_type` early-returns False for any non-`TIRScalar` (both sub-predicates gate on `isinstance(ty, tir.TIRScalar)`). ADD/SUB/MUL emit sites receive `op.results[0].ty` — a single `TIRType`. Non-scalar types (`TIRTensor` / `TIRTile` / `TIRRef` / `TIRUnit`) fall through to the 32-bit-int else branch, where the slot-of operations raise loudly on a width mismatch. The widening cannot introduce a wrong-arm dispatch.

---

## Positive observations (no finding)

- **No source drift since cycle-103**: the 26dfa82 → 31e1725 delta is pure markdown. The discipline of separating audit-doc commits from source commits preserves audit-gate integrity.
- **Cycle-103 sub-threshold notes carry forward correctly**: every below-80 observation cycle-103 disposed (test placement, local imports, edge-case coverage of SUB/MUL, sibling 64-bit width-gate sites) remains accurately classified in cycle-104. The deferred-known carve-outs are stable.
- **Stage 28.13.2 commit message self-documents the duplicated parser body**: explicit acknowledgement of the ~160 LOC duplication and deferral to future refactor — meets the cycle-19 / cycle-93 / cycle-97 audit-comment convention for "known-shape technical debt declared at landing".

---

## Verdict

**PASS** — zero findings at confidence ≥ 80 within the cycle-104 scope.

No source delta since cycle-103 baseline. Re-walked dimensions surface no new defect. Cross-stage cuts (generic-mono ↔ backend predicates, PAT_OR drain ↔ `bn_state`) integrate correctly. The deferred-known set (cycle-101 codereview F2 tail, cycle-101 silent-failures F1, cycle-57 `_is_i64_type`-only fallthrough siblings) remains stable and is not re-flagged per cycle-104 scope rules.

Stage 28.9 audit-gate counter advances **1/5 → 2/5**.

---

## Cross-reference

- **cycle 101** (`docs/audit-stage28-9-cycle101-codereview.md`): FAIL — F1 (missing regression test) + F2 (DIV/MOD/SHR signed-only).
- **cycle 102** (commit `26dfa82`): closed 3 of 4 cycle-101 findings; deferred 2 (DIV/MOD/SHR + A.StrLit).
- **cycle 103** (`docs/audit-stage28-9-cycle103-codereview.md`): PASS, 0 findings. Audit-gate counter 0/5 → 1/5.
- **cycle 104** (this doc, HEAD `31e1725`): PASS, 0 findings. Audit-gate counter 1/5 → 2/5.

---

## No code edits made

Strict read-only audit. One Write of this doc only. No Edit calls, no source mutation, no test run.
