# Audit Stage 28.9 cycle 78 — Code review

Scope: HEAD a792ee3 (cycle-77 fix-sweep — 2 backend codegen fixes: FFI_CALL
float-arg ABI split + for-range i64 increment dtype).

Mode: STRICT READ-ONLY. No edits performed. No code modified by this audit.

Narrow scope per cycle-78 charter:
- Comment vs code drift in the FFI_CALL fix and the for-range fix.
- Regression test coverage for both fixes.
- Whether the introduced fallback ("i32" default in lower_ast.py:1852) is
  clearly commented.
- Cross-walker consistency: now that FFI_CALL handles float args, do
  INDIRECT_CALL / shape_call_* arms need the same fix?

Deferred-known items from cycles C1..C77 are NOT re-flagged.

## Result: FAIL — 1 finding at conf >= 75%

## Findings

### C78-1 (HIGH, conf 82) — Cycle-77 ships 2 HIGH-conf bug fixes with zero new regression tests

Commit a792ee3 modifies `helixc/backend/x86_64.py` (FFI_CALL float-arg ABI
split) and `helixc/ir/lower_ast.py` (for-range i64 increment dtype). The
commit touches only those two source files plus three audit docs — no
files under `helixc/tests/` are added or modified. Grepping the test tree
for `extern "C" fn sinf` / `extern "C" fn` returning f32 or f64 / FFI
with float args / `for i in 0i64..N` / `range.*i64` returns no hits.

Both fixed bugs are silent-corruption (the FFI bug fed `edi` garbage to
xmm0; the for-range bug leaked 4 bytes of uninit stack on every loop
increment). Both passed typecheck + compile + the prior 1508-test heavy
gate pre-fix, which is exactly why they were latent. The cycle-77 commit
asserts post-fix heavy-gate cleanliness, but the heavy gate already
passed pre-fix, so re-running it is a necessary-but-not-sufficient
verification — there is no test in the suite whose status flips on the
fix. Without a regression test, a future refactor of the FFI_CALL arm
or the for-range lowering can silently undo either fix and the gate
will stay green.

The cycle-78 charter explicitly lists test coverage for these two
scenarios as a review axis, so this is in-scope.

Remediation (out of scope for this read-only audit): add at least one
test that compiles `extern "C" fn cosf(x: f32) -> f32` (or sinf/sqrtf)
and either inspects the emitted asm for movss-into-xmm0 or runs the
binary under WSL like `test_ffi.py` already does for `puts`; and one
test that lowers `for i in 0i64..N { ... }` and asserts the increment
op's second operand has dtype i64, or runs the binary and checks
correctness.

## Non-findings (axes reviewed but below 75% threshold)

- Comment vs code drift in the FFI fix: the new comment block at
  x86_64.py:1745-1753 accurately describes the pre-fix bug, the SysV
  class split, and the symmetry with the CALL arm. The pre-existing
  one-liner at line 1726 ("Arg shuffle is identical to CALL: int args ->
  ...") is mildly stale (omits floats), but the immediately following
  cycle-77 block fully documents the float path. Not actionable.
- Comment vs code drift in the for-range fix: lower_ast.py:1845-1851
  accurately describes the pre-fix bug and the dispatch-by-result-type
  pathology in the backend. Clean.
- "i32" fallback at lower_ast.py:1852: the conditional
  `start_v.ty.name if isinstance(start_v.ty, tir.TIRScalar) else "i32"`
  picks i32 when the iterator type is not a scalar. The block comment
  above explains the bug but not *why* i32 is the safe fallback or
  which non-scalar TIRTypes ever reach a for-range start position.
  Concern level ~60% — borderline, below the 75% bar.
- Cross-walker consistency: searched the entire backend for
  `INDIRECT_CALL`, `SHAPE_CALL`, `shape_call`, `IndirectCall`. No
  matches. The only CALL-family OpKinds in `tir.OpKind` reachable from
  `x86_64.py` are `CALL` (already handles float args correctly; the
  cycle-77 commit message confirms this and lines 1682-1707 show it)
  and `FFI_CALL` (now fixed). No other call arm needs the float split.
  Non-issue.

## Action

Cycle 78 returns FAIL with 1 HIGH finding at confidence 82. Counter
resets per the Stage 28.9 protocol.
