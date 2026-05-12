# Audit Stage 28.9 cycle 82 — Code review

**Scope:** HEAD `7b13010` (Stage 28.9 cycle-81 fix-sweep: 2 cycle-80 findings — test discrimination).

**Mode:** STRICT READ-ONLY. Read/Grep/Glob/Bash only. ONE Write (this doc). NO Edit.

**Out-of-scope:** Stage 28.10 commits in git log; deferred-known issues from prior C1–C81; areas covered by sibling audits (silent-failures, type-design).

---

## Cycle-81 changes audited

Diff `7b13010~1..7b13010`:

- `helixc/tests/test_ffi.py`: +62 / -30 — rewrote `test_c76_1_ffi_call_routes_f32_args_to_xmm0`
  to build a CONTROL int-FFI ELF alongside the float-FFI ELF and compare movss
  byte counts (`float_load > int_load`, `float_store > int_store`). Caller fn signature
  changed from `fn entry(x: f32) -> f32` to `fn caller() -> i32` so no float bytes
  leak from the caller's own prologue/epilogue.
- `helixc/tests/test_ir.py`: 1-byte change — body literal `1_i64` → `7_i64` so the
  search for `CONST_INT(value=1, ty=i64)` can only match the for-range increment.
- `docs/audit-stage28-9-cycle80-*.md`: 3 new audit findings docs (cycle-80 work product).

---

## Review

### 1. FFI test comparison — clear and self-documenting?

The rewritten `test_c76_1_ffi_call_routes_f32_args_to_xmm0` is well-engineered:

- 24-line docstring explicitly names cycle-81 C80-1 and explains BOTH the
  discrimination defect (movss bytes from the caller's own frame contaminated
  the byte-pattern assertion) and the two fixes (control program + non-float
  caller signature).
- The two source programs (`float_src` and `int_src`) are minimal and parallel;
  the only meaningful difference is the float-vs-int FFI invocation.
- Movss-load (`F3 0F 10`) and movss-store (`F3 0F 11`) constants are commented
  with their SysV semantic meaning.
- Both assertion failure messages cite both the float-program and int-program
  counts, so a regression diagnoses itself from CI output without needing the
  developer to re-run with prints.

No findings.

### 2. i64 test `7_i64` value change — comment / docstring updated?

The body literal in `test_c76_f1_for_range_i64_increment_dtype_matches_iterator`
changed from `total += 1_i64` to `total += 7_i64`. The discrimination logic now
relies on this change: `CONST_INT(value=1, ty=i64)` can only originate from the
for-range increment because the body literal contributes `value=7` instead.

The docstring (lines 160–166) still only describes the cycle-77 pre-fix dtype
bug; it does not mention cycle-81 C80-2 or the reason the body literal was
changed. However, the per-line comment at lines 178–180 already explains the
lookup discrimination strategy ("the increment-step one is the one whose
result_ty matches the iterator (i64) and value=1"), and the use of `7_i64` in
the source is self-explanatory once a reader runs `git blame` on the cycle-81
commit (which carries a thorough explanation in its message).

Assessment: borderline doc-completeness concern, ~70% confidence — below the
75% threshold. The information IS recoverable from (a) the inline comment,
(b) the git commit message, and (c) the obvious-by-inspection fact that the
test body uses `7_i64` while the search filters on `value == 1`. Not flagging.

### 3. Cycle-77/79/81 leftover clean-up

- Old single-ELF FFI test body fully replaced; no commented-out fragments
  or dead helpers left behind in `test_ffi.py`.
- `_build_and_run` (cycle 16.5 era) still used by neighboring tests; not
  cycle-77/79/81 scope.
- `helixc/backend/x86_64.py` FFI_CALL arm split-by-class (cycle 77) and
  float-return arm (cycle 79) are out-of-scope for code-review at cycle 82
  (covered in their own cycle audits and already-clean cycles).
- `helixc/ir/lower_ast.py` for-range increment dtype propagation (cycle 77)
  out-of-scope.

No findings.

---

## Verdict

**PASS** — 0 findings at confidence ≥ 75%.

One sub-threshold observation (i64 test docstring does not explicitly cite
cycle-81 / explain the `7_i64` choice in prose, ~70% confidence) noted but
not flagged.

No edits performed; this audit is read-only with a single Write to this doc.
