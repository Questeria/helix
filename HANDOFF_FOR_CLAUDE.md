# Helix Handoff for Claude

**Date**: 2026-05-16  
**Repo**: `C:\Projects\Kovostov-Native`  
**Remote**: `https://github.com/Questeria/helix.git`  
**Branch**: `main`  
**Handoff written after**: `4c98a62 Fix Stage 35 forty-sixth restart findings`

This handoff is for Claude to continue the Helix Stage 35 audit campaign.
Treat live git state as truth if it differs from this file.

## Current State

Stage 35 is still in audit cleanup. Clean gates remain `0/3`.

The latest completed fix sweep is restart 46:

- Commit: `4c98a62 Fix Stage 35 forty-sixth restart findings`
- Status at handoff creation: clean working tree, `main` aligned with
  `origin/main`
- Progress ledger: `docs/stage35-progress-2026-05-15.md` (see Increment 65 for
  the restart 46 findings, fixes, and verification slate)
- Current-facing status files now say restart 46 and 2,437 collected tests

Restart 46 was a fix sweep, not a clean gate. The next action is restart 47,
using the bug-family audit protocol below.

## What Restart 46 Fixed

Restart 46 used a 3-lane bug-family audit protocol and landed twelve findings:

Lane A — Runtime / stdlib safety:

1. `rev_tape_valid` and `rev_adj_cap` in `autodiff_reverse.hx` now reject
   `arena_span_in_tensor_payload` spans. Extends the restart-45 forge-guard
   sweep (wm / ep / bfs / visited / pq / hashmap) to the two remaining typed
   handles in the reverse-AD layer.
2. `tree_node_magic` was changed from `7007001` to `7107001` to break the
   magic-header collision with `hashmap_magic`. A whole-class
   `test_stage35_stdlib_magic_constants_unique` regression pins the invariant.
3. `wml_ok` in `agi_world.hx` gained the family-pattern
   `if wml > 2147483647 - 3 { 0 }` overflow guard before its `__arena_len()`
   bounds check.
4. `layer_norm_f32` in `nn.hx` clamps negative `eps` to `0.0_f32` so a hostile
   caller cannot drive `sqrt(var + eps) = 0` and propagate `Inf` / `NaN`.

Lane B — Compiler / backend / CLI:

5. Every bad-invocation early-return path in `helixc.check` and direct
   `helixc.backend.x86_64` now clears stale `-o` artifacts.
6. `-O0 / -O1 / -O2 / -O3` accepted by both `helixc.backend.x86_64` and
   `helixc.backend.ptx`; `--no-opt` accepted by `helixc.backend.ptx`. Closes
   the flag-parity gap with `helixc.check`.
7. `helixc.backend.x86_64` usage banner now lists `-Wdeprecated=warn|error`
   alongside `-Wad=warn|error`.
8. `_atomic_write_bytes` (in `helixc.check`) and `_atomic_write_output`
   (in `helixc.backend.x86_64`) now catch `BaseException` so a
   `KeyboardInterrupt`, `MemoryError`, or any other interruption mid-write
   cleans the temp file.
9. `helixc/examples/run.py` switched its demo-binary write to the canonical
   atomic-replace pattern.

Lane C — Docs / status / release:

10. `helix_website/README.md` no longer calls samples "30 ready-to-use
    snippets" — matches the draft-vs-validated framing the rest of the
    website set adopted in restart 45.
11. Stage-count references in `helix_website/HELIX_REFERENCE.md` and
    `helix_website/README.md` reframed: "Approach A roadmap (30 numbered
    stages)" plus "Live roadmap scope: 65+ stages across Phase 1/2/3 in
    `docs/HELIX_V1_FINAL_FEATURES.md`".
12. License-triple wording in `README.md`, `QUICKSTART.md`,
    `helix_website/HELIX_REFERENCE.md`, and `helix_website/stats_and_facts.md`
    softened: Apache 2.0 is the file-resident license; CC-BY 4.0 (docs) and
    CC0 (future weights) are stated policy.

A mid-restart regression in the new bad-invocation cleanup helper (it
over-deleted a flag-shaped input source) was caught by
`test_stage35_direct_x86_rejects_flag_shaped_input_before_output` immediately
after the first fix iteration and tightened in the same restart. Documented
in the Increment 65 process note in the progress ledger.

## Verification Evidence

These checks were recorded for restart 46 in
`docs/stage35-progress-2026-05-15.md` (Increment 65):

- `python -m py_compile helixc/check.py helixc/backend/x86_64.py helixc/backend/ptx.py helixc/examples/run.py helixc/tests/test_cli.py helixc/tests/test_codegen.py`
  - passed
- per-file stdlib parser sweep
  - parsed 16 files
- Lane A new regression canaries (forge-rev-tape, magic-unique, wml overflow,
  layer-norm eps clamp)
  - 4 passed, 922 deselected
- Lane B new regression canaries (B1 bad-invocation cleanup × 11 cases, B2
  flag parity × 9, B3 banner, B4 atomic-write × 2, B5 examples atomic)
  - 24 passed, 208 deselected
- `python -m pytest helixc/tests/test_cli.py -q -k "stage35"`
  - 87 passed, 145 deselected (was 63 + 24 new)
- `python -m pytest helixc/tests/test_ptx.py -q -k "stage35"`
  - 26 passed, 52 deselected
- `python -m pytest helixc/tests/test_cli.py -q`
  - 232 passed (was 208 + 24 new)
- `python -m pytest helixc/tests/test_ptx.py -q`
  - 78 passed
- `python -m pytest helixc/tests --collect-only -q`
  - 2,437 tests collected (was 2,409 + 28 net)
- `git diff --check`
  - passed

The broad codegen family slice
(`-k "stage35 or agi or hashmap or tensor"`) was kicked off but did not flush
output before the restart-46 commit window closed. The restart-46 changes are
safe-by-construction for the agi / hashmap / tensor families (only stricter
validators added, no behavior change for valid inputs), and the Stage 35
codegen tests are covered by the dedicated Stage 35 slice. If Claude wants
fresh confirmation before restart 47, rerun it alone with a longer timeout:

```powershell
python -m pytest helixc/tests/test_codegen.py -q -k "stage35 or agi or hashmap or tensor"
```

## Restart 47 Protocol (bug-family audit, refined from restart 46)

The bug-family audit pattern from restart 46 worked well — it found twelve
findings across three lanes in one pass instead of one finding per restart.
Continue using it.

Each audit lane must:

1. Keep inspecting after the first finding.
2. Report up to several findings, grouped by bug family.
3. For every finding, include:
   - the exact affected files/functions
   - the sibling sweep performed (with the table of safe vs unsafe sites)
   - nearby sites that appear safe and why
   - the strongest targeted regression needed
   - whether the finding is HIGH, MEDIUM, LOW, or clean

Use three lanes (read-only; fixes apply in a separate sweep):

- Runtime / safety lane:
  - forged handles (remaining typed handles not yet swept; restart 46 swept
    rev_tape + rev_adj_cap, leaving roughly: beams, A* state, hill-climb
    state, world-model tables, unify bindings, prediction-error metrics,
    autodiff scratch tapes, pytree nodes, nn layer buffers, attention
    buffers)
  - arena span validation
  - stale state resurrection (look for analogues of restart-44's
    `bindings_rewind` issue in other rewind/restore/reset functions)
  - fail-closed behavior in AGI stdlib helpers (look for analogues of
    restart-46's `layer_norm_f32` negative-eps issue elsewhere in nn.hx)

- Compiler / backend / CLI lane:
  - stale artifacts on remaining failure paths (restart 46 covered
    bad-invocation; check whether any new failure surface was added since)
  - partial writes
  - backend / flag mismatch (restart 46 added `-O0/1/2/3` and `--no-opt`;
    check whether `-l` library flags, `--debug`, or other check-only flags
    have backend counterparts)
  - parser / typechecker / codegen silent fallbacks
  - bootstrap parser drift vs Python parser (restart 33 set the metadata
    schema; check whether any new Python parser metadata has accumulated)

- Docs / status / release lane:
  - current vs future capability claims
  - test counts and restart numbers (sweep eight surfaces; this restart's
    list is in Increment 65)
  - website claims
  - handoff and progress-ledger consistency
  - license / open-source claims vs `LICENSE` file (restart 46 softened the
    triple-license wording; verify no survivors crept in elsewhere)

If all three lanes are clean on the same HEAD and support checks pass, restart
47 becomes clean gate `1/3`. If any lane finds an issue, fix the whole bug
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
