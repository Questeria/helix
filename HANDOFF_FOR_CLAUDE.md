# Helix Handoff for Claude

Date: 2026-05-23
Repo: `C:\Projects\Kovostov-Native`
Remote: `https://github.com/Questeria/helix.git`
Branch: `main`

This handoff replaces the old Stage 35 takeover file. Treat live git state as
truth if it differs from this file.

## Current Snapshot

Helix is in v3.0, Phase E, after Stage 213 chunk C. The latest pushed code
commit before this handoff work was:

```text
ff497af6b99f7a4d620000d957857adac1106eb3 mlir: add backend pipeline runner contract
```

`main` was fetched from `origin` on 2026-05-23 and was aligned with
`origin/main` before the handoff-doc update. The compiler worktree is
intentionally dirty with an MLIR audit-hardening batch. Do not commit or push
the dirty compiler changes yet.

## What Is Safe To Commit/Push Now

Only status and handoff material is safe to publish:

- `HANDOFF_FOR_CLAUDE.md`
- `docs\V3_HANDOFF.md`
- `docs\HELIX_MLIR_AUDIT_PACKET.md`

The compiler/test changes below are local working-tree state and remain
audit-blocked:

- `helixc\ir\mlir\backends.py`
- `helixc\ir\mlir\validate.py`
- `helixc\tests\test_mlir_backends.py`
- `helixc\tests\test_mlir_validate.py`
- `scripts\mlir_audit_canaries.py` (local untracked restart helper)

If you are taking over in the same machine/worktree, use those local dirty
files. If you are taking over from a fresh GitHub clone only, ask Anthony for
the local working-tree copy or reconstruct the batch from
`docs\HELIX_MLIR_AUDIT_PACKET.md`.

## Verification Refreshed On 2026-05-23

These commands were rerun from `C:\Projects\Kovostov-Native` before this
handoff:

```powershell
python scripts\mlir_audit_canaries.py --strict
python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q
python -m compileall helixc\ir\mlir helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py scripts\mlir_audit_canaries.py
python -m pytest -k mlir -q
git diff --check -- docs/V3_HANDOFF.md docs/HELIX_MLIR_AUDIT_PACKET.md helixc/ir/mlir/backends.py helixc/ir/mlir/validate.py helixc/tests/test_mlir_backends.py helixc/tests/test_mlir_validate.py scripts/mlir_audit_canaries.py HANDOFF_FOR_CLAUDE.md
```

Results:

- strict MLIR canaries: `12 passed / 0 failed`
- focused validator/backend tests: `274 passed`
- compileall: clean
- MLIR pytest slice: `410 passed, 4347 deselected`
- diff-check: only LF-to-CRLF warnings, no whitespace errors

These green gates do not make the batch committable because the third audit
round is still blocked.

## Current Audit Status

The current source of truth is `docs\HELIX_MLIR_AUDIT_PACKET.md`, section:

```text
2026-05-22 Stop Checkpoint After Third Audit Round
```

Third audit round status:

- Silent-failure axis: BLOCKED.
- Type-design axis: BLOCKED.
- General-review axis: stopped at Anthony's request before completion.

Open HIGH findings:

1. Control predicates are underchecked. Invalid `scf.if`, `cf.cond_br`, and
   `cf.assert` predicates can still pass when the predicate SSA value is not
   `i1`.
2. Memref access semantics are underchecked. `memref.load` and
   `memref.store` still need rank/index arity checks, index operand type
   checks, load result element checks, and store value type checks.
3. Constants/vector/loop semantics are underchecked. Examples include
   `arith.constant true : i32`, `arith.constant 1 : f32`, non-index
   `scf.for` bounds/steps, invalid `vector.transfer_read`,
   invalid `vector.shape_cast`, and invalid `vector.multi_reduction`.
4. Generic function bodies can bypass canonical terminator/static checks.
   Examples include generic `func.func` spellings that hide a missing
   terminator from smoke-aware validation.

Open must-fix MEDIUM findings:

1. Generic `llvm.func` input symbol binding is skipped in one backend path.
2. LLVM typed-value validation can accept scalar constants for aggregate or
   vector returns, e.g. `ret { i32 } 0` and `ret <4 x i32> 0`.
3. HIP/MSL C-like preflight can still accept impossible declarations or
   statements such as `float * 123;` and `this * is * nonsense;`.

## Required Next Move

Do one bounded audit/fix increment. Do not start Stage 214 yet.

1. Confirm the local worktree still matches this state:

   ```powershell
   cd C:\Projects\Kovostov-Native
   git status --short --branch
   git log -1 --oneline
   ```

2. Run the local canary gate:

   ```powershell
   python scripts\mlir_audit_canaries.py --strict
   ```

3. Pick one finding family, preferably control predicates or memref access.
   Fix the family and sibling sites together. Add or extend canaries/tests for
   the exact invalid examples.

4. Run the gate ladder:

   ```powershell
   python scripts\mlir_audit_canaries.py --strict
   python -m pytest helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py -q
   python -m pytest -k mlir -q
   python -m compileall helixc\ir\mlir helixc\tests\test_mlir_validate.py helixc\tests\test_mlir_backends.py scripts\mlir_audit_canaries.py
   git diff --check -- docs/V3_HANDOFF.md docs/HELIX_MLIR_AUDIT_PACKET.md helixc/ir/mlir/backends.py helixc/ir/mlir/validate.py helixc/tests/test_mlir_backends.py helixc/tests/test_mlir_validate.py scripts/mlir_audit_canaries.py HANDOFF_FOR_CLAUDE.md
   ```

5. Re-run all three clean audit axes from scratch:

   - silent-failure hunt
   - type-design analysis
   - general code review

   Each reviewer must keep inspecting after the first issue and report all
   HIGH and must-fix MEDIUM findings. If any HIGH or must-fix MEDIUM remains,
   verify it, fix the whole family, and rerun all gates and all three axes.

6. Commit only after the strict canaries, tests, compileall, diff-check, and
   all three audit axes are clean. Use explicit path staging. Push after the
   commit.

## Audit Discipline Anthony Requires

- Fail closed always. Unsupported compiler constructs must raise or return a
  FAILED validation result, never emit plausible-but-wrong output.
- Per chunk: audit the diff on all three axes.
- At stage close: audit whole touched files holistically.
- At phase close: run the five-clean gate across frontend, IR, backend,
  runtime, and tests.
- Never force-push to `main`.
- Never skip git hooks.
- Never use broad `git add .`; stage explicit paths only.
- Do not commit the current dirty compiler batch until the re-audit loop is
  clean.

## GitHub State

The intended GitHub update for this handoff is a docs-only commit containing
the current takeover documents. The dirty compiler/test changes are not pushed
as code because they are audit-blocked.

After the docs-only handoff commit, `origin/main` should contain this file and
the current audit packet, while the local worktree should still show the dirty
MLIR compiler/test files.

## Telegram

Anthony expects concise Telegram updates after meaningful progress or blockers.

Use:

```powershell
python C:\Projects\Kovostov\runtime\lib\kovostov_telegram.py send --chat 8212106071 --msg "Helix update: <short status>. Next: <next step>."
```

## Important Project Docs

- `docs\V3_PLAN.md` - v3.0 implementation plan and shipped chunk history.
- `docs\V3_HANDOFF.md` - current v3 continuation notes.
- `docs\V3_STAGE210_MLIR_DECISION.md` - ratified MLIR decision.
- `docs\POST_V3_ROADMAP.md` - v4 to v9 roadmap, only after v3.0 is complete.
- `docs\HELIX_MLIR_AUDIT_PACKET.md` - current audit packet and restart source.

## One-Sentence Takeover

You are taking over a local, tested, audit-blocked MLIR hardening batch on v3
Stage 213. Finish one open audit family, rerun the full gate ladder and all
three audit axes, then commit/push only when clean.
