# Helix MLIR Audit Acceleration Packet

Date: 2026-05-21

Purpose: make the next Helix v3.0 MLIR audit/fix restart faster without
weakening the three-clean audit rule.

## Current Repo State

The working tree is intentionally dirty with unresolved Stage 213 MLIR
audit work. Do not commit or push until the next fix batch is green and
the three audit axes report 0 HIGH and 0 must-fix MEDIUM.

Dirty files at packet creation:

- `helixc/ir/mlir/validate.py`
- `helixc/ir/mlir/backends.py`
- `helixc/tests/test_mlir_validate.py`
- `helixc/tests/test_mlir_backends.py`

Additive accelerator files from this packet:

- `docs/HELIX_MLIR_AUDIT_PACKET.md`
- `scripts/mlir_audit_canaries.py`

Last green gates before the latest re-audit findings:

- `python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q`
  - `189 passed`
- `python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_toolchain.py helixc\tests\test_mlir_mapping.py helixc\tests\test_mlir_emit.py helixc\tests\test_mlir_backends.py -q`
  - `324 passed`
- `python -m pytest -k mlir -q`
  - `325 passed, 4347 deselected`

## Fast Startup

Run these before doing any more code work:

```powershell
git status --short --branch
python scripts\mlir_audit_canaries.py
```

The canary script defaults to report-only. Once a fix batch is expected
to close all listed families, run:

```powershell
python scripts\mlir_audit_canaries.py --strict
```

## Open Finding Families

Fix these by family, not one isolated repro at a time.

Latest canary status from the 2026-05-21 22:52Z heartbeat:

- PASS: `canonical-func-return-missing-ssa`
- PASS: `canonical-scf-if-missing-ssa`
- PASS: `generic-func-signature-correspondence`
- PASS: `quoted-symbol-preservation`
- still open: fake-validator bad type / `arith.addf` over `i32`,
  and GPU backend symbol binding

Audit note from the same heartbeat: the canonical control-op family was
extended to cover sibling SSA preflight holes for `scf.for`, `scf.while`,
`cf.assert`, `func.call`, memref access, arithmetic operands, duplicate
block labels, block-argument leakage, malformed loop region-argument
assignment lists, and multi-result SSA group references (`%0#1`). The
focused validator tests pass, but this packet is not committable until
all remaining canaries and the re-run three-clean audit are clean.

Latest quoted-symbol update from the 2026-05-21 23:23Z heartbeat:
custom `func.func @"foo/bar"` interface extraction now preserves the
quoted symbol instead of collapsing it to `_quoted_symbol`; backend
symbol extraction returns `("foo/bar",)`. The `quoted-symbol-preservation`
canary passes, `python -m pytest helixc\tests\test_mlir_validate.py
helixc\tests\test_mlir_backends.py -q` reports 217 passed, and
`python -m pytest -k mlir -q` reports 353 passed. The re-audit found
quoted-delimiter sibling holes; those were fixed by making symbol and
property parsing string-aware, avoiding `func.func` injection from
string literals, parsing pipe-containing interface symbols from the
right, and parsing quoted LLVM symbols without stopping at punctuation
inside the quote. No commit yet because strict canaries still fail the
fake-validator and GPU symbol families.

1. Fake or broken `mlir-opt` can still mint `MLIRValidation.PASSED`.
   Known repro shapes:
   - `arith.constant 1 : bananas`
   - `arith.addf` over `i32`

2. Canonical terminator/control-op operands are not fully preflighted.
   Known repro shapes:
   - `func.return %missing : i32`
   - `func.return 1 : i32`
   - `scf.if %missing { ... }`
   - sibling candidates: `scf.yield`, `scf.for`, `scf.while`,
     `scf.condition`, `cf.br`, `cf.cond_br`

3. Generic `func.func` signatures can change during correspondence.
   Status: closed by the 2026-05-21 22:52Z heartbeat validator update;
   keep the canary in strict mode.
   Known repro:
   - input generic func has `function_type = () -> i32`
   - output custom func becomes `func.func @f() { return }`

4. Backend output identity is not bound for non-LLVM targets.
   Known repro:
   - MLIR defines `func.func @expected`
   - PTX emits `.entry totally_wrong()`

5. Quoted MLIR symbols can collapse in backend correspondence.
   Status: closed by the 2026-05-21 23:23Z heartbeat validator/backend
   test update; keep the canary in strict mode.
   Known repro:
   - `func.func @"foo/bar"()`
   - symbol extraction should preserve `foo/bar`, not `_quoted_symbol`

## Fix-Batch Rule

When fixing one family, grep and handle siblings before re-auditing.

Examples:

- If adding `func.return` operand validation, also inspect
  `scf.yield`, `scf.condition`, `scf.if`, `scf.for`, `scf.while`,
  `cf.br`, and `cf.cond_br`.
- If adding PTX symbol binding, also decide the binding strategy for
  ROCm HIP, Metal MSL, and WGSL before declaring the backend family
  closed.
- If changing generic property canonicalization, add canaries for both
  generic-to-custom success and generic-to-custom semantic drift.

## Gate Ladder

Use this order after each fix batch:

```powershell
python scripts\mlir_audit_canaries.py --strict
python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q
python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_toolchain.py helixc\tests\test_mlir_mapping.py helixc\tests\test_mlir_emit.py helixc\tests\test_mlir_backends.py -q
python -m pytest -k mlir -q
python -m compileall helixc\ir\mlir helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py
git diff --check -- helixc/ir/mlir/validate.py helixc/ir/mlir/backends.py helixc/tests/test_mlir_validate.py helixc/tests/test_mlir_backends.py scripts/mlir_audit_canaries.py docs/HELIX_MLIR_AUDIT_PACKET.md
```

`git diff --check` may report LF-to-CRLF warnings in this repo; treat
actual whitespace errors as blockers.

## Three-Clean Audit Prompt Shape

Every audit prompt should include:

> Do not stop at the first finding. Sweep the full diff and all sibling
> sites in the same bug family. Return every HIGH and must-fix MEDIUM
> with concrete repro/evidence. Ignore nitpicks. Do not fix files.

Run the three axes in parallel:

- silent-failure hunt
- type-design analysis
- general code review

If any HIGH or must-fix MEDIUM remains, verify it, fix the whole family,
rerun the gate ladder, then rerun all three axes from scratch.

## 2026-05-21 23:23Z Heartbeat Checkpoint

Status: code/tests updated but not committed. Strict canaries still block the
commit gate.

Closed in this heartbeat:

- quoted `func.func` symbol/interface correspondence, including escaped quoted
  symbols, delimiter-bearing interface fields, generic `sym_name`, public
  visibility normalization, result-list normalization, function attributes,
  and string/location payload edge cases.
- LLVM IR shape-probe sibling bugs found by the audit loop: quoted identifiers
  in params, typed parameter attrs, pointer-only attrs, return attrs, named and
  single-field structs, invalid value/type identifiers, memory op attrs,
  function-tail attrs, and call argument/tail validation.
- static MLIR preflight holes found by audit: `func.return` type mismatches,
  `func.call` callee signature mismatches, call result arity, known-op result
  arity, dominated block SSA in linear CFGs, function type attributes, call
  `loc(...)`, and bare-operand normal attribute dictionaries.

Latest verified gates:

- `python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q`
  -> `240 passed`
- `python -m pytest -k mlir -q` -> `376 passed, 4347 deselected`
- `python -m compileall helixc\ir\mlir helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py scripts\mlir_audit_canaries.py`
  -> clean
- `git diff --check -- ...` -> clean except LF-to-CRLF warnings
- `python scripts\mlir_audit_canaries.py` -> `4 passed / 3 failed`
- `python scripts\mlir_audit_canaries.py --strict` -> failed on the same
  three known-open canaries

Known remaining blockers:

- fake-validator bad-type canary
- fake-validator addf-i32 canary
- GPU/non-LLVM backend symbol binding canary (`@expected` -> wrong PTX entry)
- final audit also flagged broader non-LLVM target shape predicates as loose;
  handle this with the GPU backend symbol-binding family rather than mixing it
  into the quoted-symbol/interface chunk.

## Commit/Push Rule

Only commit after:

- `scripts/mlir_audit_canaries.py --strict` is clean;
- the MLIR gate ladder is green;
- all three audit axes report 0 HIGH and 0 must-fix MEDIUM.

Use explicit path staging. Push after the commit. Send a concise
Telegram progress update only after real progress or a stop/blocker.

## 2026-05-22 01:16Z Heartbeat Checkpoint

Status: uncommitted, tested checkpoint. Do not commit yet.

Closed or materially advanced in this heartbeat:

- GPU/non-LLVM backend symbol binding canary now passes. PTX wrong-entry output
  reports a missing PTX entry for `expected`.
- Backend symbol binding is now target-aware for PTX, ROCm/HIP, Metal MSL, and
  WGSL output symbols.
- Added/updated coverage for ROCm `amdgpu_kernel` filtering, PTX `.entry` vs
  `.func` masking, `gpu.func` kernel inputs, generic `"gpu.func"` attrs,
  WGSL exact compute/workgroup attrs, PTX return params, byte-array params,
  and PTX callable/param token validation.

Latest verified gates from this heartbeat:

- `python scripts\mlir_audit_canaries.py` -> `5 passed / 2 failed`
- `python scripts\mlir_audit_canaries.py --strict` -> fails only the two
  fake-validator canaries
- `python -m pytest helixc\tests\test_mlir_backends.py -q` -> `94 passed`
- `python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q`
  -> `254 passed`
- `python -m pytest -k mlir -q` -> `390 passed, 4347 deselected`
- `python -m compileall helixc\ir\mlir helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py scripts\mlir_audit_canaries.py`
  -> clean
- `git diff --check -- ...` -> clean except LF-to-CRLF warnings

Known remaining blockers:

- strict canary: fake-validator bad type
- strict canary: fake-validator `arith.addf` over `i32`
- final GPU-family audit still has verified live blockers:
  - WGSL accepts malformed parameter names such as `fn expected(???: i32)`.
  - WGSL accepts attributed params without an identifier, e.g.
    `fn expected(@builtin(global_invocation_id) : vec3<u32>)`.
  - PTX valid `.func` signatures using `.reg` return params are currently
    false-rejected.
  - PTX valid `.func ... .noreturn { ... }` is currently false-rejected.

Restart protocol:

1. Run `python scripts\mlir_audit_canaries.py` to confirm the state is still
   `5 passed / 2 failed`.
2. Fix the verified live GPU-family blockers above or, if choosing to close
   the strict canaries first, start with the fake-validator family.
3. Re-run backend tests, focused validator/backend tests, the MLIR slice,
   compileall, diff-check, and all three audit axes before any commit.

## 2026-05-22 Stop Checkpoint After Audit Fix Batch

Status: uncommitted, tested checkpoint. The user asked to stop at a good point,
so do not start a new development tier from here.

Closed in this batch:

- Strict MLIR audit canaries are now clean: fake-validator bad type,
  fake-validator `arith.addf` over `i32`, canonical missing SSA,
  generic interface correspondence, GPU backend symbol binding, and quoted
  symbol preservation all pass.
- WGSL now rejects malformed parameter names and attributed parameters without
  identifiers.
- PTX `.func` now accepts `.reg` return/parameter forms and `.noreturn`, while
  rejecting entry-only directives on functions and malformed predicate guards.
- Static MLIR preflight now rejects unsupported obvious function types,
  duplicate `func.func` symbols, empty returns from non-void functions, and
  same-line declaration boundary drift.
- Backend shape probes now reject the reproduced malformed LLVM/WGSL/HIP/MSL
  artifacts and return no symbols for malformed target text.
- Validator/backend tool identity is tighter: direct MLIR validation must use
  a path matching a fresh support probe, and backend lowering requires the
  lowering tool path to match validation provenance.

Verified so far after the fix batch:

- `python scripts\mlir_audit_canaries.py --strict` -> `7 passed / 0 failed`
- `python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q`
  -> `266 passed`
- `python -m pytest -k mlir -q` -> `402 passed, 4347 deselected`
- `python -m compileall helixc\ir\mlir helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py scripts\mlir_audit_canaries.py`
  -> clean
- `git diff --check -- ...` -> clean except LF-to-CRLF warnings

Still required before commit:

- Re-run all three audit axes from scratch. The previous audit round was
  BLOCKED and these fixes have not yet been re-audited.

If resuming from here, start with the remaining gate ladder and re-audit. Do not
commit until all three audit axes report 0 HIGH / 0 must-fix MEDIUM.

## 2026-05-22 Stop Checkpoint After Third Audit Round

Status: uncommitted, tested, audit-blocked checkpoint. The user asked to stop
at a good point. Do not commit or push this batch yet.

Closed since the previous stop checkpoint:

- LLVM backend shape now rejects empty/label-only function bodies and
  no-terminator definitions before symbol binding.
- LLVM `alloca` parsing is comma-aware: valid `alloca i32, align 4` accepts,
  malformed trailing garbage rejects.
- `arith.cmpf` / `arith.cmpi` result typing preserves vector/tensor `i1`
  shape instead of recording scalar `i1`.
- Static MLIR preflight now rejects canonical missing function terminators and
  illegal vector dimensions such as `vector<?xi32>`, `vector<0xi32>`,
  `vector<*xi32>`, and `vector<[0]xi32>`, while allowing 0D
  `vector<f32>`.
- HIP/MSL/WGSL backend shape checks now reject malformed C-like params,
  obvious `@` statement junk, WGSL invalid `alias`/`type`/`var` declarations,
  ROCm HIP top-level LLVM instruction crumbs, and scalar typed-value
  impossibilities such as `ret i32 true`.
- Fast audit canaries were expanded from 7 to 12 cases to include missing
  terminators, vector bad dims, LLVM typed-value shape, HIP malformed params,
  and WGSL malformed declarations.

Verified before the third audit round:

- `python scripts\mlir_audit_canaries.py --strict` -> `12 passed / 0 failed`
- `python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q`
  -> `274 passed`
- `python -m pytest -k mlir -q` -> `410 passed, 4347 deselected`
- `python -m compileall helixc\ir\mlir helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py scripts\mlir_audit_canaries.py`
  -> clean
- `git diff --check -- ...` -> clean except LF-to-CRLF warnings

Third audit round status:

- Silent-failure axis: BLOCKED.
- Type-design axis: BLOCKED.
- General-review axis: stopped by user before completion; rerun it after the
  next fix batch.

Open HIGH findings to fix next:

- Fake/smoke-aware `mlir-opt` can still mint `MLIRValidation.PASSED` for
  invalid control predicates: `scf.if %c`, `cf.cond_br %c`, and `cf.assert %c`
  pass with `%c: i32` / `%c: f32`. Add predicate type checks for these control
  ops.
- Fake validator can still pass invalid memref accesses: rank/index arity,
  index operand type, load result element type, and store value type are not
  enforced for `memref.load` / `memref.store`.
- Fake validator can still pass invalid constants and vector/loop semantics:
  examples include `arith.constant true : i32`, `arith.constant 1 : f32`,
  `scf.for` bounds/steps with non-`index` types, `vector.transfer_read` with
  wrong element/index types, `vector.shape_cast vector<4xi32> to vector<3xi32>`,
  and `vector.multi_reduction <bogus>`.
- Generic function bodies can bypass terminator/static checks, e.g.
  `module { func.func @f() { "test.op"() : () -> () } }` and generic
  `"func.func"` forms can still be passed by a smoke-aware echo validator.

Open must-fix MEDIUM findings:

- Backend symbol binding skips generic `llvm.func` inputs; generic
  `"llvm.func"` with `sym_name = "expected"` can allow an unrelated LLVM
  artifact to evade symbol binding.
- LLVM typed-value shape still accepts scalar constants for aggregate/vector
  returns, e.g. `ret { i32 } 0` and `ret <4 x i32> 0`.
- HIP/MSL C-like artifact predicates still accept impossible declarations or
  statements such as `float * 123;` and `this * is * nonsense;`.

Restart protocol:

1. Confirm the worktree is still this uncommitted MLIR batch with
   `git status --short --branch`.
2. Re-run `python scripts\mlir_audit_canaries.py --strict`.
3. Fix the open findings as families, and add canaries/tests for each family.
4. Re-run focused validator/backend tests, MLIR slice, compileall, and
   diff-check.
5. Re-run all three audit axes from scratch. Commit only after all axes report
   0 HIGH / 0 must-fix MEDIUM.

## 2026-05-24 Checkpoint — Third-Round HIGH Fix Batch (Claude)

Status: progress checkpoint. Three of the four open HIGH families from the
2026-05-22 Third-Audit-Round stop are now closed in the static preflight;
HIGH-4 (generic function bodies bypass) remains open. The MLIR slice and the
strict canaries are green; the new code was 3-clean audited and the one HIGH
finding the audit raised was fixed.

Closed since the previous stop:

- HIGH-1 (control predicates) — `scf.if`, `cf.cond_br`, `cf.assert` now
  require an `i1` predicate. New helper `_control_predicate_type_finding`
  looks the first SSA operand up in `ssa_types` and rejects anything but
  `i1`. Two new strict canaries:
  `control-predicate-scf-if-non-i1`, `control-predicate-cf-assert-non-i1`.
- HIGH-2 (memref access semantics) — `memref.load` / `memref.store` now
  enforce index arity (count matches the memref rank from
  `_memref_rank_from_type`) and index operand type (must be `index`). The
  result-element / stored-value type checks remain a sibling follow-up.
  Three new canaries: `memref-load-index-arity-mismatch`,
  `memref-load-non-index-idx`, `memref-store-index-arity-mismatch`.
- HIGH-3 (constants / loop semantics, dominant sub-cases) —
  `arith.constant` now matches literal-vs-type: `true`/`false` require
  `i1`, decimal integer literals (including `1_000`) require an
  integer/`index` type, float literals require a floating-point type.
  Hex/octal/binary-prefixed literals (`0x...`, `0o...`, `0b...`) defer
  (they may legitimately encode a float bit-pattern under `: f32`).
  `scf.for` now requires `index` bounds. Two new canaries:
  `arith-constant-bool-non-i1`, `arith-constant-int-float-type`. The
  vector sub-cases (`vector.transfer_read`, `vector.shape_cast`,
  `vector.multi_reduction`) remain open — see "Still open" below.

Audit fixes applied this batch (3-clean on the new code):

- `_memref_access_type_finding` used `bracket_end == -1` but
  `_matching_closer_index` returns `None` on no-match. The wrong sentinel
  silently sliced past the end of the op text and emitted a bogus arity
  finding. Fixed to `bracket_end is None`, returning a named
  unbalanced-bracket finding.
- `_arith_constant_value_type_finding` would misclassify hex/octal/binary
  literals (`0x7FC00000 : f32` was wrongly treated as float-by-presence
  of `E`). Added a short-circuit defer on `0x`/`0o`/`0b` prefixes and
  added `_` separator handling for decimal integer literals.

Verified gates this batch:

- `python scripts/mlir_audit_canaries.py --strict` -> `19 passed / 0 failed`
  (was 12 / 0; +2 control-predicate, +3 memref, +2 arith-constant)
- `python -m pytest helixc/tests/test_mlir_validate.py helixc/tests/test_mlir_backends.py -q`
  -> `274 passed`
- `python -m pytest -k mlir -q` -> `410 passed, 4347 deselected`

Still open (carried forward):

- HIGH-4: generic function bodies bypass terminator/static checks. A custom
  `func.func @f() { "test.op"() : () -> () }` is currently DEFERRED rather
  than FAILED because the generic-syntax op inside the custom body confuses
  the static preflight. Needs either generic-op-in-custom-body recognition
  or an outright rejection of generic ops in custom bodies. Probably
  requires parser work in `_func_body_findings`.
- Remaining HIGH-3 sub-cases: `vector.transfer_read` wrong element/index
  types, `vector.shape_cast` element-count mismatch, `vector.multi_reduction
  <bogus>` kind. Each is a focused per-op check similar to the closed ones.
- MEDIUM findings from the 2026-05-22 third audit are still in scope:
  generic `llvm.func` symbol-binding path; LLVM typed-value validation for
  aggregate/vector returns; HIP/MSL C-like preflight accepting impossible
  declarations.

Restart protocol from here:

1. `git status --short --branch` — confirm the same dirty MLIR worktree.
2. `python scripts/mlir_audit_canaries.py --strict` — confirm `19 / 0`.
3. Decide whether to close HIGH-4 (a meaningful parser change) or to stop
   the audit treadmill here and route the residual sub-cases through Stage
   215's real-`mlir-opt` parity gate.
4. If continuing, run the gate ladder + re-audit (three axes from scratch)
   before any further commit.
