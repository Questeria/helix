# Helix Handoff for Claude

**Date**: 2026-05-16  
**Repo**: `C:\Projects\Kovostov-Native`  
**Remote**: `https://github.com/Questeria/helix.git`  
**Branch**: `main`  
**Handoff written after**: `2f37a16 Fix Stage 35 forty-fifth restart findings`

This handoff is for Claude to continue the Helix Stage 35 audit campaign after
Anthony resets the computer. Treat live git state as truth if it differs from
this file.

## Current State

Stage 35 is still in audit cleanup. Clean gates remain `0/3`.

The latest completed fix sweep is restart 45:

- Commit: `2f37a16 Fix Stage 35 forty-fifth restart findings`
- Status at handoff creation: clean working tree, `main` aligned with
  `origin/main`
- Progress ledger: `docs/stage35-progress-2026-05-15.md`
- Current-facing status files now say restart 45 and 2,409 collected tests

Restart 45 was a fix sweep, not a clean gate. The next action is restart 46,
using the optimized audit protocol below.

## What Restart 45 Fixed

Restart 45 fixed three bug/documentation families:

1. Safe tensor payloads could forge valid-looking non-tensor AGI handles.
   Guards were added to reject tensor-payload spans for:
   - world-memory handles
   - episodic-memory handles
   - BFS queues
   - visited sets
   - priority queues
   - hashmaps

2. Failed output-producing compiles could leave stale binaries at `-o` paths.
   Cleanup was added for:
   - `python -m helixc.check ... -o out.bin`
   - direct `python -m helixc.backend.x86_64 source.hx out.bin`

3. Website/reference docs overclaimed current readiness.
   Wording now distinguishes:
   - current Python-hosted `helixc`
   - future self-hosted `kovc`
   - draft website code samples versus compiler-validated examples

## Verification Evidence

These checks were recorded for restart 45 in
`docs/stage35-progress-2026-05-15.md`:

- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - passed
- per-file stdlib parser sweep
  - parsed 16 files
- exact forged-handle/codegen canaries
  - 4 passed
- exact stale-output CLI canaries
  - 5 passed
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - 63 passed
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - 26 passed
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - 176 passed
- `python -m pytest helixc\tests\test_cli.py -q`
  - 208 passed
- `python -m pytest helixc\tests\test_ptx.py -q`
  - 78 passed
- `python -m pytest helixc\tests --collect-only -q`
  - 2,409 tests collected
- `git diff --check`
  - passed

Immediately before this handoff, the exact canaries and Stage 35 CLI/PTX slices
were rerun and passed. A parallel rerun of the broad codegen family slice timed
out in the wrapper; do not treat that as a regression. If Claude wants fresh
confirmation before restart 46, rerun it alone with a longer timeout:

```powershell
python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"
```

## Optimized Restart 46 Protocol

The previous audit rhythm was safe but too slow because lanes often found one
issue, stopped, then the next restart found a sibling issue. Restart 46 should
use bug-family auditing.

Each audit lane must:

1. Keep inspecting after the first finding.
2. Report up to several findings, grouped by bug family.
3. For every finding, include:
   - the exact affected files/functions
   - the sibling sweep performed
   - nearby sites that appear safe and why
   - the strongest targeted regression needed
   - whether the finding is HIGH, MEDIUM, LOW, or clean
4. Avoid editing files directly unless explicitly taking the fix-sweep role.

Use three lanes:

- Runtime/safety lane:
  - forged handles
  - arena span validation
  - stale state resurrection
  - fail-closed behavior in AGI stdlib helpers

- Compiler/backend/CLI lane:
  - stale artifacts
  - partial writes
  - backend mismatch
  - parser/typechecker/codegen silent fallbacks

- Docs/status/release lane:
  - current versus future capability claims
  - test counts and restart numbers
  - website claims
  - handoff and progress-ledger consistency

If all three lanes are clean on the same HEAD and support checks pass, restart
46 becomes clean gate `1/3`. If any lane finds an issue, fix the whole bug
family, add canaries, run verification, commit, push, and restart the clean
counter from `0/3`.

## Suggested First Commands

```powershell
cd C:\Projects\Kovostov-Native
git status --short --branch
git log -1 --oneline
Get-Content docs\stage35-progress-2026-05-15.md -Tail 140
```

Then run a fresh support baseline:

```powershell
python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py
@'
from pathlib import Path
from helixc.frontend import parser
files = sorted(Path("helixc/stdlib").glob("*.hx"))
for p in files:
    parser.parse(p.read_text(encoding="utf-8"), filename=str(p))
print("parsed", len(files), "stdlib files")
'@ | python -
python -m pytest helixc\tests\test_cli.py -q -k "stage35"
python -m pytest helixc\tests\test_ptx.py -q -k "stage35"
python -m pytest helixc\tests --collect-only -q
```

## Telegram Updates

Anthony wants beginner-friendly progress updates with estimated percent
complete. Send Telegram messages after meaningful progress and when a restart
begins/ends.

Use ASCII-only messages:

```powershell
python C:\Projects\Kovostov\runtime\lib\kovostov_telegram.py send --chat 8212106071 --msg "Helix update: <short beginner-friendly update>. Estimated Stage 35: about 95%."
```

## Commit Rules

- Use explicit path staging only. Do not use broad `git add .`.
- Do not revert unrelated user changes.
- Commit only after targeted canaries and relevant family tests are green.
- Push to `origin main` after a good commit.

## Important Reminder

The current production compiler is still Python-hosted `helixc`. The
self-hosted Helix compiler remains the target, not the shipped replacement yet.
Do not claim Helix is fully self-hosted until the repo proves it with
repeatable self-hosting tests.
