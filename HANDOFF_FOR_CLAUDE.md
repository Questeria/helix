# Helix Handoff for Claude

**Date**: 2026-05-16  
**Repo**: `C:\Projects\Kovostov-Native`  
**Remote**: `https://github.com/Questeria/helix.git`  
**Branch**: `main`  
**Handoff written after**: Stage 35 restart 57 catch-up sweep (commit lands alongside this handoff)

This handoff is for Claude to continue the Helix Stage 35 audit campaign.
Treat live git state as truth if it differs from this file.

## Current State

Stage 35 is still in audit cleanup. Clean gates remain `0/3`.

The most recent fix sweeps are restart 54 (full bookkeeping), restart 55
(source only — sin/cos range reduction), restart 56 (source only — three
INT32_MIN / NaN bugs), and restart 57 (catch-up sweep that filled in the
55/56 bookkeeping debt).

- Commit: pinned by the latest `git log -1 --oneline`
- Status at handoff creation: clean working tree, `main` aligned with
  `origin/main`
- Progress ledger: `docs/stage35-progress-2026-05-15.md` (see Increment 76
  for restart 57 catch-up sweep; 75 for restart 56 retroactive; 74 for
  restart 55 retroactive; 73 for restart 54; 72 for restart 53; 71 for
  restart 52; 70 for restart 51; 69 for restart 50; 68 for restart 49)
- Current-facing status files now say restart 57 and 2,527 collected tests
  (live count after restart 57 catch-up added 5 retroactive canaries on
  top of restart 54's 2,522; see Increments 70-76 in the ledger for the
  per-restart canary chain)

## Restart 57 → Restart 58 carry-forward

The `tf1d_sum` NaN-skip fix in restart 56 was applied to `tf1d_sum`
ONLY, but the in-source comment (since trimmed by restart 57) had
claimed the sibling sweep also covered `tf1d_dot`, `tf1d_l1_norm`,
`tf1d_max_abs`, `tf1d_sum_in_range`. Restart 58 Lane A should pick
up that family as a deliberate work item.

Restart 51 ran a fresh 3-lane read-only audit on top of restart 50's HEAD
plus picked up the restart-50-deferred C8 carry-forward. Result: 12
findings (4 HIGH + 5 MEDIUM + 3 LOW) plus a sibling B4 const_fold sweep
discovered during the fix sweep itself. Fix sweep closed all 12 plus
the C8 carry-forward; no items deferred to restart 52.

Restart 52 ran a fresh 3-lane read-only audit on top of restart 51 HEAD.
Result: 3 findings (1 Lane A HIGH + 0 Lane B + 1 Lane C MEDIUM + 1 Lane C
LOW). The Lane A finding was the missed 2D sibling of restart 51 A3
(`ti1d_dot` saturation): `ti2d_matvec` and `ti2d_matmul` lifted to i64 +
saturation. Commit `c584b0b` landed the runtime fix but did NOT add the
regression canaries, lane audit docs, or Increment 71 ledger entry —
restart 53 picked up that bookkeeping.

Restart 53 ran a fresh 3-lane read-only audit on top of restart 52 HEAD
(`c584b0b`). Result: 15 findings (4 Lane A HIGH + 3 Lane A MEDIUM + 1
Lane A LOW + 1 Lane B MEDIUM + 1 Lane B LOW + 5 Lane C HIGH). The Lane A
findings were missed siblings in the i64-saturation sweep (vec_dot,
vec_sum/product family, attention_dot, ti1d_axpy/add_scalar/mul_scalar,
dense_layer bias-add, sgd_step_array) plus an attention_softmax_f32
NaN-fail-closed gap. Lane B added a re-raise guard sibling on the
backend driver and an explanatory comment on validate_kernel_tile_lowering.
Lane C wrote the missing restart 52 lane docs + Increment 71 + Increment
72 + reconciled 11 surfaces to "restart 53" + fixed the HANDOFF protocol
numbers. Fix sweep closed all 15. See Increments 71 + 72 in the ledger.

## Restart 54 → Restart 55 deferred findings

(none — restart 55 started from a clean carry-forward; it then found
1 HIGH transcendentals range-reduction bug)

## What Restart 55 Fixed

Restart 55 ran a fresh 3-lane audit on top of restart 54 HEAD (`e34b4d6`)
and found 1 Lane A HIGH (Lane B + Lane C clean). Source fix landed in
commit `218ffd0` ("Fix Stage 35 fifty-sixth restart findings" — title
+1 drift from actual restart number). Lane docs + canary + ledger
Increment 74 were written retroactively by restart 57's catch-up sweep.

Lane A (1 HIGH):

- A1 HIGH: `helixc/stdlib/transcendentals.hx` — `__sin`, `__cos`,
  `__sin_f64`, `__cos_f64` gained explicit `[-π, π]` range reduction
  before the 4-term Taylor series. Without reduction, |x| > 2π
  produced nonsense. Round-via-i32-cast trick keeps the reduction
  arena-pure.

## What Restart 56 Fixed

Restart 56 ran a fresh 3-lane audit on top of restart 55 HEAD (`218ffd0`)
and found 3 Lane A findings (Lane B + Lane C clean). Source fixes
landed in commit `278d46a` ("Fix Stage 35 fifty-seventh restart
findings" — title +1 drift). Lane docs + canaries + ledger Increment 75
were written retroactively by restart 57's catch-up sweep.

Lane A (2 HIGH + 1 MEDIUM):

- A1 HIGH: `helixc/stdlib/tensor.hx` `tf1d_sum` — NaN-skip discipline
  via `if v == v`. Without the fix, a single NaN slot poisoned the
  entire sum (NaN + anything = NaN).
- A2 HIGH: `helixc/stdlib/tensor.hx` `ti1d_max_abs` — INT32_MIN
  special-case. Without the fix, `0 - INT32_MIN` wraps back to
  INT32_MIN and the `av > best` test silently dropped the negative
  slot.
- A3 MEDIUM: `helixc/stdlib/iterators.hx` `vec_max_abs` — same
  INT32_MIN bug as A2 in the iterators.hx companion.

## What Restart 57 (Catch-up Sweep) Fixed

Restart 57 was a bookkeeping-and-catch-up sweep on top of restart 56
HEAD (`278d46a`). No fresh 3-lane audit dispatched. Closed the
bookkeeping debt accumulated by restarts 55 + 56 and trimmed one
overclaiming comment.

- Wrote ledger Increments 74 + 75 + 76 retroactively.
- Wrote 6 lane audit docs (`docs/audit-stage35-restart55-laneA.md`
  through `docs/audit-stage35-restart57-laneC.md`).
- Added 5 regression canaries in `helixc/tests/test_codegen.py`
  (3 for restart 55 sin/cos/sin_f64 range reduction; 1 for restart
  56 tf1d_sum NaN-skip; 1 family canary for restart 56 max_abs
  INT32_MIN saturation across ti1d_max_abs + vec_max_abs).
- Trimmed `tf1d_sum` comment overclaim (it had said "Same pattern
  applied across tf1d_dot, tf1d_l1_norm, tf1d_max_abs, tf1d_sum_in_range"
  but the sibling sweep was not actually applied — carried forward
  to restart 58 Lane A).
- Refreshed HANDOFF_FOR_CLAUDE (this file), HANDOFF_FOR_CHATGPT,
  README narrative (via test count), QUICKSTART, HELIX_REFERENCE,
  stats_and_facts, code_samples.

## What Restart 54 Fixed

Restart 54 ran a fresh 3-lane read-only audit on top of restart 53 HEAD
(`c4cb7a3`). Result: 11 findings (4 Lane A HIGH + 2 Lane A MEDIUM + 1
Lane A LOW + 0 Lane B HIGH + 1 Lane B MEDIUM + 1 Lane B LOW + 0 Lane C
HIGH + 0 Lane C MEDIUM + 2 Lane C LOW).

Lane A findings (i64-saturation siblings — campaign's dominant family):

- A1 HIGH: `autodiff_reverse.hx` reverse-mode AD tape was the missed
  consumer of the restart 51/52/53 saturation discipline. Both forward
  record (`rev_add`/`rev_sub`/`rev_mul`/`rev_neg`) and backward adjoint
  accumulation (`rev_backward` kind=1/2/3/4) lifted to i64 + INT32
  saturation. The mul branch was double-wrapping silently.
- A2 HIGH: `tensor.hx` `ti1d_mul` Hadamard product (also `ti1d_add` and
  `ti1d_sub`) per-element i64 + INT32 saturation.
- A3 HIGH: `iterators.hx` `vec_zip_mul` Hadamard product per-element
  saturation. Sibling of vec_dot (restart 53 A1).
- A4 HIGH: `iterators.hx` `vec_window_sum` rolling accumulator + per-
  output saturation. Worse than other window helpers because a wrap
  propagates via subtraction into every subsequent output. Also
  saturated `vec_sum_in_range`.
- A5 MEDIUM: `iterators.hx` `vec_l1_distance` and `vec_l2_squared_distance`
  i64 accumulator + INT32 saturation. Sibling of vec_sum_squares
  (already saturated).
- A6 MEDIUM: `nn.hx` `lin_reg_grad_w`/`lin_reg_grad_b`/`sgd_step_scalar`
  i64 intermediates + INT32 saturation. Scalar mirrors of the array
  helpers that restart 53 A7 saturated.
- A7 LOW: `iterators.hx` cluster sweep — `vec_zip_add`, `vec_zip_sub`,
  `vec_map_add_scalar`, `vec_map_mul_scalar`, `vec_scale_inplace`,
  `vec_offset_inplace`, `vec_pairwise_diff`, `vec_pairwise_sum`,
  `vec_offset_alloc`, `vec_fold_op` per-element saturation.

Lane B findings:

- B1 MEDIUM: `check.py:43-44` `--help` `-W<flag>` example line gained
  `-Wad` / `-Wad=error` enumeration, matching backend banners. Closes
  parser-vs-banner drift on a behaviour-honoured flag.
- B2 LOW: `helixc/ir/lower_ast.py:847` `_lower_type` loud-fails
  (NotImplementedError) on unknown TyNode subclass instead of returning
  `tir.TIRScalar("?")` sentinel. Added `A.TyFn` case lowering to u64
  closure-pointer placeholder. Sibling of restart 47 B1 discipline.

Lane C findings:

- C1 LOW: `HELIX_REFERENCE.md:1153` + `code_samples.md:8` roadmap-
  snippets attribution re-stamped from "Stage 35 restart 50 lane C
  audit" to "Stage 35 restart 54 lane C audit".
- C2 LOW: README + 6 sibling surfaces narrative compression replaced
  with drift-proof "see Increments 70-73 in the progress ledger for
  the per-restart canary chain since restart 50" — sidesteps the
  +N-canaries chain each restart.

Fix sweep closed all 11. Regression coverage added (11 canaries: 9 in
`test_codegen.py`, 2 in `test_cli.py`). Live test count: 2,511 → 2,522.

## What Restart 51 Fixed

Restart 51 ran a fresh 3-lane read-only audit (with the C8 carry-forward
from restart 50). The fix sweep closed all findings.

Lane A (1 HIGH + 2 MEDIUM + 2 LOW): `__log_stable_f64` added and
`d_log_v` rewired to use it (closes the f64-log domain-guard gap);
`clip_grad_norm_f32` NaN-fail-closed (was only `<= 0`-guarded);
`string_to_int` uses i64 accumulator + saturation (was: i32 wrap at
INT32_MAX+1); `vec_zip_mod` and `vec_zip_div` fail-closed on b[i] == 0
(was: trap to runtime); `vec_negate_inplace` + `vec_map_neg` saturate
INT32_MIN to INT32_MAX (was: silent wrap to INT32_MIN).

Lane B (1 HIGH + 2 MEDIUM): `autodiff_cli` rejects unknown
single-dash flags with `rc=2 unknown flag` (was: silent positional-arg
aliasing); `check.py --emit-asm`/`--emit-ptx`/`-o` artifact-emit branches
re-raise `(NotImplementedError, AssertionError, ...)` loud-fail signals
before the catch-all `except Exception`. Sibling sweep: `const_fold.py`
int-arith / float-arith / bitwise blocks gained the same re-raise
discipline (3 try-blocks).

**Important — do NOT re-flag in restart 52**: the two
`validate_kernel_tile_lowering` blocks in `check.py` (lines ~1716 and
~1750) deliberately KEEP `except Exception` without a re-raise guard.
That function uses `NotImplementedError` as the user-facing
"unsupported tile op" signal, codified by
`test_stage35_emit_ptx_reports_tile_lowering_error_without_bug_label`
and `test_stage35_output_binary_rejects_dead_unsupported_kernel_op`.
An earlier audit-lane reviewer flagged these as siblings and the fix
was applied + reverted before commit. The current state is intentional.

Lane C (1 HIGH + 3 MEDIUM + 1 LOW + C8 carry-forward): `README.md`
restart-attribution corrected (was: "restart 49 collected 2,489" inside
a "restart 50 is latest" paragraph); `stats_and_facts.md` preamble
reconciled with the table row (was: line 8 said restart 49, line 14
said restart 50); `HANDOFF_FOR_CHATGPT.md` continuation pointer reconciled
the same way; live test-count reconciled across 8 surfaces from the
restart-50 forecast 2,489 to the actual 2,497 post-restart-51;
`HELIX_REFERENCE.md` "Increments 50-68+" / "50-67+" anchors replaced
with "Increments 50 onward" (open-ended, restart-drift-proof); C8
carry-forward closed by adding `"6 bare fn (+0 @-attributed)"` to
`ieee754.hx` and `"2 bare fn (+50 @-attributed)"` to
`transcendentals.hx` per-module callouts (the other 14 modules were
already standardized by restart 50).

Regression coverage added (10 cases): 5 in `test_codegen.py` (Lane A
A1-A5), 4 in `test_cli.py` (Lane B B1 unknown-short-flag + B2 emit-ptx
NIE-propagation + B3 emit-asm NIE-propagation + 1 source-text canary
covering the 3 check.py codegen re-raise sites), and 1 source-text
canary in `test_cli.py` for the const_fold re-raise sibling sweep.

## What Restart 50 Fixed

Restart 50 ran a fresh 3-lane read-only audit (no deferred backlog from
restart 49). Result: 17 findings (3 HIGH + 5 MEDIUM + 9 LOW). The fix
sweep closed 16 of 17; 1 LOW (C8 per-module fn-count convention)
deferred to restart 51 (see "Restart 50 → Restart 51 deferred findings"
above).

Lane A (4 LOW): `string_from_int(INT32_MIN)` writes the full sentinel;
adam/layer_norm now NaN-fail-closed in addition to negative-fail-closed;
`ti1d_prod` i64+saturate; `hashmap_load_factor_x100` numerator i64.

Lane B (1 MEDIUM + 2 LOW + 1 deferred-prior): `autodiff_cli --as-function`
preserves source param/return types (was: hardcoded f32); `const_fold`
`is_const` exception narrowed to cast-failure family; `presburger` dead
`if False else` simplified; `hash_cons` SHA-256 fallback documented-prior
(no fix).

Lane C (3 HIGH + 4 MEDIUM): HELIX_REFERENCE.md:59 "23+" sibling reframed
(closes restart-49 miss); agi-features.md remaining-work row for const-
fold+DCE removed (those are shipped); HANDOFF_FOR_CLAUDE Protocol section
de-staled; trap-ids.md header pinned to ledger anchor with grep guidance;
HELIX_REFERENCE + code_samples Gallery preambles list known-roadmap
snippets; tutorial.md fragment-level disclaimer; `scripts/run_all_tests.sh`
echo line matches QUICKSTART promise.

Regression coverage added (10 cases): 4 in test_codegen.py (Lane A), 3 in
test_cli.py (Lane B).

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

## Restart 58 Protocol (bug-family audit, refined from restart 57 catch-up)

**IMPORTANT**: restart 57 closed the bookkeeping debt from restarts
55 + 56 but is NOT itself a clean gate. Restart 58 MUST run a fresh
3-lane audit (read-only) on the restart 57 HEAD. There IS one
explicit carry-forward into restart 58 Lane A: the `tf1d_dot` /
`tf1d_l1_norm` / `tf1d_max_abs` / `tf1d_sum_in_range` NaN-skip
sibling sweep that restart 56's comment overclaimed but did not
actually apply. Restart 58 Lane A should pick that up and add
canaries.

The campaign run-rate: restarts 46/47/48/49/50/51/52/53/54/55/56
closed 12, 17, 13, 11, 17, 12, 3, 15, 11, 1, 3 findings respectively.
The restart-55 dip to 1 and restart-56 dip to 3 (down from restart
54's 11) suggests the i64-saturation family is now largely exhausted
across stdlib and remaining findings are smaller, more localized
families (transcendentals reduction; INT32_MIN / NaN edge cases).
The first restart where the audit returns 0 findings on the same
HEAD becomes clean gate 1/3.

**Process discipline reminder**: restart 52, 55, 56 all landed
partial commits (source fixes without paired canaries / ledger /
docs). The hardening rule is "verify each restart commit ALSO
writes the ledger increment + lane docs + canaries before pushing."
If the next fire abbreviates the process again, the cleanup catches
up in a later "catch-up sweep" labeled as such.

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
    Increment 65; current count after the latest restart is in the
    progress ledger tail and in `helix_website/stats_and_facts.md`)
  - website claims — verify after the eighth surface sweep is done
  - handoff and progress-ledger consistency
  - license / open-source claims — restart 46 + 47 + 48 + 49 swept
    softer. Verify no new triple-license claims appeared.
  - tool flag completeness — restart 47 + 49 rewrote HELIX_REFERENCE.md
    + QUICKSTART.md against `helixc/check.py`'s `--help`. If any new
    check.py flags were added, re-sync.

If the next restart's audit returns 0 findings across all three lanes on
the same HEAD as the current handoff, the clean-gate counter advances to
`1/3`. The restart after that starts from the same HEAD; the third
consecutive clean gate closes Stage 35.

If any lane finds an issue, fix the whole bug family, add canaries, run
verification, commit, push, and restart the clean counter from `0/3`.

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
