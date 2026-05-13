# Stage 31 Reset Handoff - 2026-05-13

User asked to stop at a good point so the computer can reset.

## Current State

- Worktree: `C:\Projects\Kovostov-Native`
- Broad regression tests were green before the reset:
  - `python -m pytest -q -p no:cacheprovider helixc\tests --ignore=helixc\tests\test_codegen.py`
    - Result: `1108 passed in 147.18s`
  - `python -m pytest -q -p no:cacheprovider helixc\tests\test_codegen.py`
    - Result: `742 passed in 1123.02s`
- After resume, the known findings below were fixed and the new sharded full
  validator passed:
  - `python scripts\stage31_validate.py --mode full --shards 2`
  - Non-codegen: `1108 passed`
  - Codegen shard 1: `394 passed, 348 deselected`
  - Codegen shard 2: `348 passed, 394 deselected`
  - Snapshot smoke: `EXIT:42`
- Stage 30 snapshot folder exists:
  - `C:\Projects\Kovostov-Native\HELIX_STAGE30_COMPILER_SNAPSHOT`
- Snapshot guide exists and its simple compile/run path was verified:
  - `C:\Projects\Kovostov-Native\HELIX_STAGE30_COMPILER_SNAPSHOT\AI_USAGE_GUIDE.md`
  - Verified command path produced `EXIT:42`.
- Fresh clean gate count is `0/3`.

## Important Answer To User Question

Helix is not fully Helix-only yet. The practical compiler is still Python-hosted
in `helixc`. Stage 30 proved a Helix-written bootstrap compiler path in
`helixc\bootstrap`, but full replacement requires feature parity, repeated
self-host cascades, identical binary checks, and making the Helix-built compiler
the default.

## Fresh Audit Findings Fixed After Resume

### 1. Tuple/Array Aggregate Returns Must Fail Closed

Fresh Clean Gate 1 found that this typechecks and emits bad IR:

```hx
fn make() -> (i32, i32) { (1, 2) }
```

Command:

```powershell
$p = Join-Path $env:TEMP 'stage31_agg_tuple_return_repro.hx'
Set-Content -LiteralPath $p -Value 'fn make() -> (i32, i32) { (1, 2) }' -NoNewline
python -m helixc.check $p --emit-ir -O0 --no-stdlib
Remove-Item -LiteralPath $p
```

Status: fixed. `TypeChecker._is_unsupported_aggregate_return_type` now rejects
unsupported tuple/array aggregate returns before IR lowering, similar to the
existing struct/nonrecursive enum return rejection.

### 2. `helixc.check` And Backend Default Optimization Order Differ

Fresh Clean Gate 2 found pass-order drift:

- `helixc\check.py` default `-O1`: `fdce -> const_fold`
- `helixc\backend\x86_64.py` default: `const_fold -> cse -> dce -> fdce`

Status: fixed for host IR/ELF. `helixc.check` default `-O1` now mirrors the
backend order for host IR/ELF: `const_fold -> cse -> dce -> fdce`. `--emit-ptx`
keeps DCE/FDCE off because the textual kernel body is the inspected artifact.

Minimal repro source:

```hx
type Probability = f64 where self >= 0.0_f64, self <= 1.0_f64;
fn dead() -> i32 { 99 }
fn main() -> i32 {
    let p: Probability = 0.5_f64;
    let x = 1 + 2;
    let y = 1 + 2;
    x + 39
}
```

### 3. Stage 30 Snapshot Contains Stale Operational Docs

Fresh Clean Gate 3 found packaging problems:

- `HELIX_STAGE30_COMPILER_SNAPSHOT\HANDOFF_FOR_CHATGPT.md` contains stale live
  work instructions and old commit/status references.
- `HELIX_STAGE30_COMPILER_SNAPSHOT\README.md` is stale at Stage 28.9 and points
  to missing docs.

Status: fixed. The snapshot handoff was replaced with a frozen snapshot notice,
and the snapshot README now points to the guide and Stage 30.1 evidence only.

## Speed-Up Added

New scripts:

- `scripts\pytest_shard.py`: stable hash-based pytest sharding.
- `scripts\stage31_validate.py`: quick/full validation runner. Full mode runs
  non-codegen plus sharded `test_codegen.py` in parallel, then snapshot smoke.

## Do Not Forget

- Send Telegram updates using:

```powershell
python C:\Projects\Kovostov\runtime\lib\kovostov_telegram.py send --chat 8212106071 --msg "<message>"
```

- Keep language beginner-friendly with progress percentages.
- Do not stage broad/unrelated files. Use explicit paths only.
- Old untracked audit docs are unrelated unless the user asks.
- After any code/doc fix, reset clean gate count and rerun 3 fresh clean gates.
