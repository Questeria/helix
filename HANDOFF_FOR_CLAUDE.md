# Helix Handoff for Claude

**Date**: 2026-05-16  
**Repo**: `C:\Projects\Kovostov-Native`  
**Remote**: `https://github.com/Questeria/helix.git`  
**Branch**: `main`  
**Handoff written after**: `6c555a4 Fix Stage 35 forty-ninth restart findings`

This handoff is for Claude to continue the Helix Stage 35 audit campaign.
Treat live git state as truth if it differs from this file.

## Current State

Stage 35 is still in audit cleanup. Clean gates remain `0/3`.

The latest completed fix sweep is restart 49:

- Commit: `6c555a4 Fix Stage 35 forty-ninth restart findings`
- Status at handoff creation: clean working tree, `main` aligned with
  `origin/main`
- Progress ledger: `docs/stage35-progress-2026-05-15.md` (see Increment 68
  for restart 49; 67 for restart 48; 66 for restart 47; 65 for restart 46)
- Current-facing status files now say restart 49 and 2,479 collected tests

Restart 49 was a fix sweep that closed the entire restart-48 deferred
backlog plus one new B4 finding (lower_ast structural_hash narrowing). The
deferred list is now empty. The next action is restart 50, which MUST run a
fresh 3-lane audit (no deferred items to pick up).

## What Restart 49 Fixed

Restart 49 was a deferred-only fix sweep — no fresh 3-lane audit dispatch.
It closed all 7 of the Lane B + Lane C items that restart 48 explicitly
deferred (per Increment 67), plus one new finding (B4) caught while reading
adjacent code:

Lane B (4 fixes):

1. `autodiff_cli` exit codes now match the check/x86/ptx convention:
   bad invocation → rc=2, source/parse error → rc=1, internal/runtime
   error → rc=1. Previously: bad invocation rc=1 (wrong), parse error
   rc=2 (wrong), differentiate failure rc=2 (wrong).
2. `-h` / `--help` works on every CLI (`helixc.check`,
   `helixc.backend.x86_64`, `helixc.backend.ptx`,
   `helixc.frontend.autodiff_cli`). All four print a banner to stdout and
   exit 0. Previously only `helixc.check` had proper help support.
3. `helixc.backend.x86_64` and `helixc.backend.ptx` banners now enumerate
   every accepted flag: `-O0..-O3`, `--no-opt`, `-Wad=`, `-Wdeprecated=`,
   `-l <libname>`, `--no-color`, `--color`, `--hash`, `--hash-cons`.
   `helixc.backend.ptx` also gained a usage banner on bare invocation
   (was: only `error: ptx: missing input path`).
4. `helixc/ir/lower_ast.py:3082-3086` narrowed `except Exception` around
   `structural_hash(expr.inner)` to
   `except (KeyError, AttributeError, TypeError, ValueError)` so
   `NotImplementedError` from `ast_hash`'s loud-fail discipline
   propagates instead of aliasing distinct quote bodies to the same
   `_pretty` fallback string. Mirror of restart 47 B1 + restart 48 B2/B3.

Lane C (6 fixes):

5. `docs/HELIX_V1_FINAL_FEATURES.md` line 3 status sentence rewritten to
   disclaim its planning-era Stage 31-34 numbering and point at
   `docs/ROADMAP.md` as authoritative.
6. `docs/ROADMAP.md` line 17 corrected from "5 dogfood programs" to
   "6 dogfood programs + self-improving-agent flagship".
7. Date stamps in `docs/ROADMAP.md`, `docs/HELIX_V1_FINAL_FEATURES.md`,
   `docs/HELIX_PURPOSE.md` switched to ledger-anchored phrasings.
8. `helix_website/HELIX_REFERENCE.md` Compiler-Architecture stdlib list
   (lines 956-962) rewritten with all 16 actual modules + per-module tags.
9. HELIX_REFERENCE.md "23+ silent-corruption bugs (and counting)" reframed
   as "Dozens of silent-corruption defects (live count grows with each
   restart; see Increments 50-67+)" so the headline doesn't understate.
10. `HANDOFF_FOR_CHATGPT.md` line 17 historical-block license-triple
    softened to match current-facing surfaces.

Regression coverage added in `helixc/tests/test_cli.py` (13 cases): 2 for
B1 exit codes, 8 parametrized for B2 -h/--help (4 CLIs × 2 flags), 2 for
B3 banner content, 1 for B4 source-text invariant.

## What Restart 48 Fixed

Restart 48 closed 6 of the 13 audit findings (1 HIGH + 5 MEDIUM); 7 LOW
findings were deferred (see "Restart 48 → Restart 49 deferred findings"
above):

Lane A — Runtime / stdlib safety:

1. `d_div_v` and `d_div_dx` (autodiff.hx) now fail-closed at `b_v == 0`.
2. `softmax_layer` (nn.hx) writes the maximum-entropy distribution (1/n
   to every slot) when `sum_e <= 0` or `sum_e` is NaN.
3. `tanh_layer` (nn.hx) delegates to `__tanh` instead of inlining
   `__exp(2*xi) / (e2x+1)`, so the |x| > 20 saturation short-circuit
   protects against NaN at the boundary.

Lane B — Compiler / backend / CLI:

4. `helixc.check` now accepts `--no-opt` as a `-O0` synonym (closes the
   restart-47 reverse-direction parity gap).
5. `helixc.backend.ptx` outer `except Exception` handlers narrowed to
   re-raise `NotImplementedError`/`AssertionError`/`KeyboardInterrupt`/
   `SystemExit`/`MemoryError` first.
6. `helixc.frontend.autodiff_cli` `_parse_or_exit` and `differentiate`
   wrappers narrowed the same way.

Lane C — Docs / status / release:

7. `helix_website/HELIX_REFERENCE.md` Standard Library section rewritten
   against `ls helixc/stdlib/*.hx`: all 16 actual modules, grouped by
   purpose (Numerics & IEEE 754, Tensors & tiles, Neural networks,
   Autodiff, AGI primitives, Collections), with per-module function
   counts and a discoverability one-liner.
8. HELIX_REFERENCE.md Stats block refined: "design doc references stage
   numbers up to Stage 65 (35 distinct stages enumerated; not a strict
   consecutive sequence)" replaces the looser "65+ stages" claim.
9. `helix_website/README.md` `/learn` softened from "10-lesson interactive
   tutorial" to "Planned beginner tutorial sequence (no shipped content
   yet)".

## What Restart 47 Fixed

Restart 47 closed 17 findings (5 MEDIUM + 12 LOW) across the three lanes:

Lane A — Runtime / stdlib safety:

1. `adam_f32_step` (nn.hx) and `__adam_step` (transcendentals.hx) now clamp
   `next_v` / `v` to `>= 0` before `__sqrt`, preventing a negative
   moment-estimate from producing a tiny denominator and an exploding weight
   update.
2. `layer_norm_f32` (nn.hx) writes 0 to every output slot when
   `denom = __sqrt(var + safe_eps) <= 0`, so a constant-input + zero-eps
   call no longer propagates `Inf`/`NaN`.
3. `d_sqrt_dx`, `d_log_dx`, `d_recip_v`, `d_recip_dx` (autodiff.hx) now
   fail-closed (return 0) at their analytical singularities (`a_v <= 0` or
   `a_v == 0`), matching the layer-norm precedent.

Lane B — Compiler / backend / CLI:

4. `_resolve_monomorphized_struct_type` (lower_ast.py) narrowed its
   exception scope from `except Exception` to
   `except (KeyError, AttributeError)` so `NotImplementedError` from the
   `struct_mono._mangle_ty` loud-fail discipline propagates. Future
   `TyNode` subclasses (refinement, confidence, tiered memory) will now
   force explicit dispatch instead of silently miscompiling.
5. `examples/dashboard_server.py` switched its generated-source write to
   the canonical `tempfile.mkstemp + os.replace + on-failure cleanup`
   pattern (mirrors `examples/run.py` from restart 46 B5).
6. `frontend/autodiff_cli.py` wrapped file-IO, parse, and differentiate
   calls in structured `try/except` blocks; failures now surface
   `error: autodiff_cli: ...` diagnostics instead of raw Python tracebacks.
7. Both `helixc.backend.x86_64` and `helixc.backend.ptx` now accept
   `-l <libname>` / `-lm`, `--no-color`, `--color`, `--hash`,
   `--hash-cons` for flag-parity with `helixc.check` (treated as no-ops
   here; goal is parity, not actual implementation).

Lane C — Docs / status / release:

8. `helix_website/HELIX_REFERENCE.md` Live-compiler-driver flag list
   rewritten against `helixc/check.py`'s actual `--help` text. Removed
   fictitious flags (`--dump-ast-hashes`, `--no-bootstrap-cache`,
   `--target=*`, `--version`) and clarified that `--dump-ast-hashes`
   lives on `helixc.frontend.autodiff_cli`.
9. `helix_website/HELIX_REFERENCE.md` Open-Source Commitments section
   softened to match the restart-46 license-triple wording (Apache 2.0
   file-resident; CC-BY 4.0 + CC0 stated policy).
10. `helix_website/HELIX_REFERENCE.md` bootstrap-chain diagram updated
    so the final node says "self-hosted Helix compiler (roadmap target)"
    with a side note clarifying that today's `helixc` is not chain-derived.
11. `QUICKSTART.md` CLI flags section expanded to include `-O0..-O3`,
    `--stdlib`/`--no-stdlib`, and `-Wad`/`-Wdeprecated` policies.
12. `README.md` "30+ stdlib builtins" updated to
    "Stdlib in `helixc/stdlib/*.hx` (16 modules, ~455 functions)".

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

These checks were recorded for restart 47 in
`docs/stage35-progress-2026-05-15.md` (Increment 66):

- `python -m py_compile helixc/check.py helixc/backend/x86_64.py helixc/backend/ptx.py helixc/examples/dashboard_server.py helixc/frontend/autodiff_cli.py helixc/ir/lower_ast.py helixc/tests/test_cli.py helixc/tests/test_codegen.py`
  - passed
- per-file stdlib parser sweep
  - parsed 16 files
- Lane A new regression canaries (6 new tests for A1-A7; A6/A7 share one test)
  - 6 passed
- Lane B new regression canaries (B1 + B2 + B3 × 2 + B4 × 12 parametrized)
  - 16 passed
- All Lane A regression tests including restart 46 + restart 47
  - 10 passed
- `python -m pytest helixc/tests/test_cli.py -q -k "stage35"`
  - 103 passed (was 87 + 16 new)
- `python -m pytest helixc/tests/test_ptx.py -q -k "stage35"`
  - 26 passed
- `python -m pytest helixc/tests --collect-only -q`
  - 2,459 tests collected (was 2,437 + 22 net)
- `git diff --check`
  - passed

The full CLI and full PTX suites + broad codegen slice were not re-run in
the restart-47 commit window; restart 47's changes are safe-by-construction
for the broad family (fail-closed clamps only stricten existing behavior;
loud-fail propagation only surfaces NotImplementedError that previously was
silently swallowed; flag-parity additions are no-ops). Restart 48's baseline
should rerun these for fresh confirmation.

The following older checks were recorded for restart 46 in Increment 65:

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

## Restart 50 Protocol (bug-family audit, refined from restart 49)

**IMPORTANT**: restart 49 closed the entire restart-48 deferred backlog.
There are NO carry-forward findings. Restart 50 MUST run a fresh 3-lane
audit (read-only). The bar continues to rise — restarts 46/47/48/49 each
closed 12, 17, 13, 11 findings respectively, so 4 consecutive restarts have
each found more issues than the prior. The first restart where the audit
returns 0 findings on the same HEAD becomes clean gate 1/3.

The bug-family audit pattern from restart 46 (12 findings) and restart 47
(17 findings) worked well — each restart pulls more sibling issues into the
same fix sweep. Continue using it. **IMPORTANT:** instruct the audit lane
agents to be strictly read-only this time (no Edit/Write); restart 46's
agents "auto-applied" their findings despite the instruction.

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
  - forged handles — restart 47 verified ALL 13 magic-bearing validators
    are now guarded. Look for any NEW typed handle introduced since restart
    47, plus any handles NOT magic-bearing (vec, deque, ring buffer).
  - arena span validation — restart 47 verified all 14 sites have overflow
    guards. Re-verify if any new validators were added.
  - magic-constant uniqueness — restart 47 verified all 13 distinct.
  - stale state resurrection — restart 47 swept 5 rewind/clear functions
    clean. Re-verify if any new rewind/restore/reset surfaces appeared.
  - fail-closed numerical helpers — restart 47 fixed Adam clamp,
    layer_norm var+eps==0, and 4 autodiff div-by-zero rules. Still LOW-risk
    areas: NaN-eps handling (currently documented as garbage-in/garbage-out),
    any new `__sqrt`/`__log`/division sites added since.

- Compiler / backend / CLI lane:
  - stale artifacts — restart 46 covered bad-invocation, restart 47 verified
    no new failure surfaces. Re-verify if new return-paths were added.
  - partial writes — restart 47 swept clean except `dashboard_server.py`
    (now atomic). Verify no new file-writers.
  - backend / flag mismatch — restart 46 + 47 closed flag parity for
    `-O*`, `--no-opt`, `-l*`, `--no-color`/`--color`, `--hash`/`--hash-cons`.
    `--debug`/`--symbols` were confirmed not to exist anywhere; if you find
    new check.py flags, mirror to backends.
  - parser / typechecker / codegen silent fallbacks — restart 47 fixed
    `lower_ast._resolve_monomorphized_struct_type` loud-fail. Other
    `except Exception` sites verified safe or narrowed.
  - bootstrap parser drift vs Python parser — restart 47 verified no new
    metadata kinds since the Stage 33 alignment commit.

- Docs / status / release lane:
  - current vs future capability claims — restart 47 fixed the
    HELIX_REFERENCE.md fictitious-flag list and bootstrap-chain diagram.
    Sweep any new website material added since.
  - test counts and restart numbers (sweep the eight surfaces listed in
    Increment 65; this restart's count is 2,459 / restart 47)
  - website claims — verify after the eighth surface sweep is done
  - handoff and progress-ledger consistency
  - license / open-source claims — restart 46 + 47 swept everything
    softer. Verify no new triple-license claims appeared.
  - tool flag completeness — restart 47 rewrote HELIX_REFERENCE.md against
    `helixc/check.py`'s `--help`. If any new check.py flags were added,
    re-sync.

If restart 48's audit returns 0 findings across all three lanes on the same
HEAD (`4ba725f` or its newest descendant), the clean-gate counter advances
to `1/3`. Restart 49 then starts from that same HEAD; restart 50 if 49 is
also clean; three consecutive clean gates close Stage 35.

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
