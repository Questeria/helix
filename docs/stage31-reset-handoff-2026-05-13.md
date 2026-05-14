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
- Fresh clean gate count is `3/3` after the proof/source-read and validation
  harness fixes.

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
- `scripts\run_all_tests.sh`: legacy "run everything" entry point now delegates
  pytest coverage and snapshot smoke to `stage31_validate.py --mode full`,
  bootstraps a local `.stage31-venv` if Bash's Python lacks pytest, uses a
  `.stage31-bin/wsl` compatibility shim when already inside WSL, then still
  runs the `stage0/hex0` bootstrap-floor gate directly from the current Bash.
  Follow-up speedup: the default full gate now auto-selects `min(cpu_count, 8)`
  codegen shards, while `HELIX_TEST_SHARDS=N` still overrides it. This keeps
  the same tests and hash-based one-shard-per-test coverage; it only increases
  safe parallelism on machines with enough cores.
  Verification: `bash scripts/run_all_tests.sh` auto-selected 8 shards on the
  16-core development machine and passed pytest, snapshot smoke, and
  `stage0/hex0` in about 506 seconds.

## 2026-05-13 Follow-Up Increment

Bundled the first AGI-safe scalar refinement aliases into the default stdlib:

- `Confidence = f64 where 0.0 <= self <= 1.0`
- `Probability = f64 where 0.0 <= self <= 1.0`
- `DistanceMeters = f64 where self >= 0.0`

These names now work in ordinary Helix programs without local redefinition
when the default stdlib is enabled. `--no-stdlib` still requires the program to
define or import these names explicitly.

Audit follow-up: default stdlib merge now treats `type`, `struct`, and `enum`
as one type-name namespace for user-vs-stdlib conflicts. This prevents a user
`struct Probability` or `enum Confidence` from silently coexisting with and
being shadowed by the bundled stdlib aliases.

## 2026-05-13 Proof Artifact Slice

Added `python -m helixc.check --emit-proof-obligations <file.hx>`.

The flag emits JSON with schema `helix.proof_obligations.v0`. Stage 31 records
refinement obligations as:

- `proved`: the constant checker proved the predicate.
- `failed`: the checker proved violation and reports trap `31001`.
- `unproven`: the value is not compile-time-proven yet; future SMT/runtime
  proof support is needed.
- `unsupported`: the predicate shape is outside the Stage 31 constant checker.

Normal progress lines go to stderr in this mode so stdout remains parseable
JSON for tooling. Stdout-producing modes are mutually exclusive; combining
`--emit-proof-obligations` with `--doc`, `--emit-ast`, `--emit-ir`,
`--emit-asm`, or `--emit-ptx` is a bad invocation with rc=2.
For nested aliases, unproven values emit obligations for both the outer
refinement and inherited base refinements, so future SMT tooling sees the full
constraint set.

Validation reliability follow-up: `helixc/tests/test_codegen.py::compile_and_run`
now writes unique temp ELF names instead of naming only by ELF hash. Many
different tests compile to byte-identical "return 42" binaries, and sharded
pytest workers could collide while WSL was chmod/execing those files. The
failed shards passed after this harness fix, and the full wrapper gate passed.

## 2026-05-13 Proof Artifact Metadata Slice

Proof-obligation JSON now includes an `input` block:

- `source_sha256`: SHA-256 of the exact UTF-8 source bytes read by `helixc.check`.
- `include_stdlib`: whether default stdlib was merged.
- `stdlib_manifest_sha256` and `stdlib_files`: deterministic hashes for every
  stdlib file included in the proof input.
- `opt_level`, `flags`, `libs`, and `warnings`: normalized CLI settings.
- `color`: normalized diagnostic color mode (`auto`, `always`, or `never`).

This gives future proof caches and audit tooling a stable key tying the proof
artifact back to the source and compiler invocation.

Audit follow-up: proof-mode parse diagnostics are rendered colorless inside the
JSON artifact so `--color` cannot inject ANSI escapes into machine-readable
messages. AD warnings drained after proof typecheck now appear in
`warning_diagnostics`, with `summary.warning_errors` tracking `-Wad=error`
promotions that make the command fail even when typecheck itself succeeded.
Second audit follow-up: proof artifacts also classify missing stdlib-file
warnings, deprecated warnings promoted by `-Wdeprecated=error`, totality
warnings promoted by `--strict`, and strict effect-check warnings. This prevents
stdout JSON from looking clean when stderr or the process exit code says the
invocation failed.
Third audit follow-up: proof mode now keeps JSON output for strict missing
stdlib failures and mirrors the normal hard validation passes for trace, panic,
unwind, unsafe, and autotune errors before declaring a proof artifact clean.
Fourth audit follow-up: proof input flags now drop the explicit `--stdlib`
compatibility no-op so default stdlib and explicit stdlib produce the same
proof key. CLI warning policies are validated up front; typoed values like
`-Wdeprecated=erro` are rejected as bad invocations instead of silently
demoting a requested warning promotion.
Fifth audit follow-up: CLI warning names are also validated (`ad` and
`deprecated` are the current supported names), so typos like
`-Wdeprectaed=error` cannot silently fail to promote. Proof mode now still
collects strict effect-check warning records even when hard validation
pipeline errors, such as trace errors, are already present.
Sixth audit follow-up: strict proof-mode effect checking now traps its own
lowering failures into a `strict-effect-check` pipeline error, preserving JSON
stdout for invalid source shapes like malformed `panic()` that were already
reported by hard validation.
Seventh audit follow-up: strict missing-stdlib proof errors use stable relative
stdlib names in JSON instead of absolute checkout paths. Source decode failures
in proof mode now emit a JSON artifact with `phase: "decode"` and the source
hash instead of returning stderr-only.
Eighth audit follow-up: proof-mode bad invocations with a real source path
now emit `phase: "invocation"` JSON artifacts for invalid warning names or
policies. Strict effect-check now catches failures from optimization/effect
passes, not just lowering, and reports them as `strict-effect-check` pipeline
errors instead of escaping to the generic internal-error handler.
Ninth audit follow-up: proof-mode stdout conflicts, missing source paths, and
missing source files now also emit `phase: "invocation"` JSON artifacts instead
of empty stdout, help text, or stderr-only failures.
Verification after this follow-up: focused proof-invocation tests passed, full
`test_cli.py` passed with 113 tests, `test_typecheck.py` passed with 229 tests,
and `bash scripts/run_all_tests.sh` passed with 8 auto-selected codegen shards,
snapshot smoke, and `stage0/hex0` in about 507 seconds.

## 2026-05-13 Validation Harness Override Fix

Manual shard overrides are now capped at 8. The full gate still auto-selects
up to 8 shards by default, but unreasonable values such as
`HELIX_TEST_SHARDS=999999` fail immediately with rc=2 instead of trying to
launch impossible numbers of pytest workers.
Verification: `test_stage31_validate.py` covers the cap and fallback behavior,
the combined CLI/typecheck/validator focused suite passed with 345 tests, and
`bash scripts/run_all_tests.sh` passed again with 8 shards, snapshot smoke, and
`stage0/hex0` in about 575 seconds.
Audit follow-up: `run_all_tests.sh` now validates `HELIX_TEST_SHARDS` before
starting pytest setup or `stage0/hex0`, so invalid overrides exit immediately
with rc=2 at the wrapper level too.
Verification after the wrapper follow-up: the Bash-side invalid override check
returned rc=2 before gates, the combined CLI/typecheck/validator suite passed
with 346 tests, and `bash scripts/run_all_tests.sh` passed with 8 shards,
snapshot smoke, and `stage0/hex0` in about 536 seconds.
Proof source-read follow-up: proof mode now emits `phase: "source-read"` JSON
when the supplied source path exists but cannot be opened as a readable file,
such as when the path is a directory.
Verification after the source-read follow-up: the directory-path proof repro
returned parseable JSON with `phase: "source-read"`, the combined
CLI/typecheck/validator suite passed with 347 tests, and
`bash scripts/run_all_tests.sh` passed with 8 shards, snapshot smoke, and
`stage0/hex0` in about 845 seconds.
Wrapper parsing follow-up: `run_all_tests.sh` now parses `HELIX_TEST_SHARDS`
in base 10 before range checks, so zero-padded values like `09` fail cleanly
before pytest or `stage0/hex0` starts.
Verification after the wrapper parsing follow-up: `HELIX_TEST_SHARDS=09`
returned rc=2 before gate banners, the combined CLI/typecheck/validator suite
passed with 348 tests, and `bash scripts/run_all_tests.sh` passed with 8
shards, snapshot smoke, and `stage0/hex0` in about 855 seconds.

## 2026-05-14 Proof Cache Key Slice

Proof-obligation JSON now emits a top-level `cache_key` derived from canonical
JSON over `{schema, input}`. This makes the artifact's proof input key explicit
for future verification gates and cache tooling. The key is path-independent
for readable source files because `path` is not part of the hashed payload.
Artifacts whose source bytes are unavailable, such as missing or unreadable
source paths, report `cache_key: null` instead of pretending to be cacheable.

Added `scripts\proof_artifact_key.py` to recompute or check a proof artifact's
cache key. The helper accepts UTF-8 and PowerShell redirected UTF-16 JSON so
Windows users can pipe artifacts without knowing shell encoding details. Stage
31 quick validation now includes the path-independence regression for proof
cache keys.
Audit follow-up: stdin now uses the same UTF-8/UTF-16 decoding path as file
input, so `proof_artifact_key.py --check -` works with PowerShell-style piped
artifacts too. Tests include an independent golden SHA-256 assertion for the
canonical `{schema,input}` payload.
Verification after this follow-up: focused proof/cache-key tests passed, the
helper checked a real PowerShell-redirected artifact through stdin, Stage 31
quick validation passed, the combined CLI/typecheck/validator/helper suite
passed with 356 tests, and `bash scripts/run_all_tests.sh` passed with 8
shards, snapshot smoke, and `stage0/hex0` in about 800 seconds.
Clean-audit gate after the stdin follow-up: 3/3 passed at high confidence.

## 2026-05-14 Proof Artifact Validator Slice

Added `scripts\proof_artifact_validate.py` to validate proof-obligation JSON
artifacts before they are trusted as audit evidence. The validator checks the
top-level schema and `cache_key`, recomputes the proof cache key, validates
summary counts, diagnostic entry shapes, proof-obligation shape, source hashes,
required input metadata, stdlib manifest consistency, and Windows-friendly
UTF-8/UTF-16 artifact loading through the shared proof-artifact loader.

Audit follow-ups made the validator stricter in the places that matter most:
malformed diagnostic sections are rejected, required proof-input metadata is
enforced, stale source files are caught automatically when the artifact path can
be resolved, relative artifact paths are resolved from the artifact JSON
directory, missing embedded source paths fail closed, JSON booleans are rejected
for integer fields, stdlib manifest hashes are recomputed from `stdlib_files`,
malformed missing-stdlib entries are rejected, source-unavailable artifacts must
carry an explicit `cache_key: null`, and source-unavailable artifacts cannot
carry proof obligations or typecheck diagnostics.

Stage 31 quick validation now includes high-signal proof-artifact validator
regressions. The snapshot smoke gate also now runs `python -m helixc...` from
the scratch directory while pointing `PYTHONPATH` at
`HELIX_STAGE30_COMPILER_SNAPSHOT`, so it verifies the frozen Stage 30 snapshot
instead of accidentally importing the live compiler from the repo root.

Verification after this slice: focused validator/key/stage31 tests passed, the
Stage 31 quick gate passed, the combined CLI/typecheck/validator/key suite
passed with 374 tests, and `bash scripts/run_all_tests.sh` passed with 8
shards, snapshot smoke, and `stage0/hex0` in about 899 seconds. Clean-audit
gate after the final source-unavailable and snapshot-isolation fixes: 3/3
passed at high confidence.

## Do Not Forget

- Send Telegram updates using:

```powershell
python C:\Projects\Kovostov\runtime\lib\kovostov_telegram.py send --chat 8212106071 --msg "<message>"
```

- Keep language beginner-friendly with progress percentages.
- Do not stage broad/unrelated files. Use explicit paths only.
- Old untracked audit docs are unrelated unless the user asks.
- After any code/doc fix, reset clean gate count and rerun 3 fresh clean gates.
- If an audit gate stalls or times out, replace it with a fresh audit instead
  of waiting endlessly.
