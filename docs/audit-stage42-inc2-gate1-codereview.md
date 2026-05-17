# Stage 42 Inc 2 Gate-1 — Code Review
Date: 2026-05-17
Scope: git diff 6f818e4..HEAD (Stage 42 Inc 0/1 ship)
HEAD: 7699f00769891faeef3196b17d8fd03f4335b58f

## Verdict
GATE CLEAN

## Findings (HIGH / MEDIUM / LOW, with confidence 0-100)

### LOW — Header comment overclaims "all 5 semantic-type families" (confidence 85)

File: `C:/Projects/Kovostov-Native/helixc/examples/dogfood_15_agi_planning_loop.hx`, lines 3-5

The header comment states:
> "AGI planning-loop scenario exercising all 5 semantic-type
>  families (memory + spatial + temporal + modal + causal) in a
>  single coherent program."

But the code does NOT exercise the memory tier (no `into_episodic`, `consolidate`, `recall`, or any `EpisodicMem`/`SemanticMem`/`WorkingMem` wrapper appears). Only 4 of 5 families are touched: spatial (`WorldFrame`), temporal (`Present`/`Future`), modal (`Known`/`Believed`), causal (`Cause`/`Effect` + `propagate`).

The corresponding progress doc `docs/stage42-progress-2026-05-17.md` lines 36-38 is internally consistent — it explicitly notes "omitting the memory wrapper which would make it 6 levels deep but adds no demonstration value beyond the 5 already exercised" (although that doc's "5 already exercised" wording is itself slightly confused since after omitting memory you have 4 wrappers + 1 causal transition = arguably "5 things" but not "5 wrapper families"). The `run.py` registration title ("4-deep wrapper stack (Stage 42 Inc 1)") is accurate.

This is purely a documentation accuracy issue inside the `.hx` source comment. It does not affect compilation, runtime behavior, the witness, or the self-host cascade.

Severity: LOW. Suggested fix: amend lines 3-5 to read "exercising 4 of 5 semantic-type families (spatial + temporal + modal + causal; memory tier deferred — would add a 5th wrapper layer without further demonstration value)" and optionally mention the causal transition `propagate` as the 5th orthogonal axis.

## Verification steps performed

- Confirmed HEAD is `7699f00` (Stage 43 Inc 0 ledger opening) and Stage 42 commits in range are: `4e74244` (open), `1e58862` (hotfix), `8c30b76` (close).
- Working tree state at audit start contained Stage 43 work-in-progress (autodiff.py, autodiff_reverse.py, typecheck.py, test_stage43_cleanup.py); the orchestrator's `gate-2-audit-stash` (stash@{0}) holds those changes. Working tree was returned clean to HEAD for the audit re-run.
- Verified all 14 constructor/eliminator names used by dogfood_15 are registered in `helixc/frontend/typecheck.py`: `into_known`, `from_known`, `into_present`, `from_present`, `into_world`, `from_world`, `into_cause`, `from_cause` (wait — dogfood uses `from_effect` not `from_cause`, both registered), `into_believed`, `into_future`, `from_future`, `from_believed`, `from_effect`, `propagate`.
- Verified `propagate` signature: `Cause<T> -> Effect<T>` (typecheck.py line 4004, 4018-4019). Matches dogfood line 62.
- Verified `into_world(T) -> WorldFrame<T>` preserves arbitrary inner type including wrapper-typed inners (typecheck.py line 3442 `inner=arg_tys[0]`). The Stage 43 idempotency check at line 3431 only rejects `TyFrame` inners; `TyCausal` (`Cause<i32>`, `Effect<i32>`) inners pass through. Confirms dogfood lines 52, 64 compile.
- Verified `from_world(WorldFrame<T>) -> T` returns the inner type unchanged (line 3459 `return arg_tys[0].inner`). Matches dogfood line 50 (`from_world(wf): i32`) and line 70 (`from_world(wf3): Effect<i32>`).
- Verified F1 cross-causal-launder `inner_is_shadowed` short-circuit is intact at HEAD: two occurrences in typecheck.py at lines 3738 and 3938 (consistent with the hotfix commit 1e58862 message which claimed restoration).
- Ran the dogfood: `python -m helixc.examples.run planning` → exit code 42. Confirms (a) typechecking succeeds at HEAD, (b) IR identity-lowering preserves the i32 value through all 4-deep wrapper layers and the `propagate` transition, (c) all three input values (10, 14, 18) round-trip correctly.
- Ran self-host gate: `python scripts/stage33_selfhost_gate.py` → G2..G4 byte-identical sha `a6f1ee44eb4418ba296954528d05564f5a37627dc38bb350b2308675d86b8986`. Matches the hotfix commit body claim exactly.
- Verified DEMOS dict in run.py has no duplicate keys. "planning" key is fresh; insertion order places it last (correct for `--list` display).
- Searched the new .hx file for `TODO`, `FIXME`, `XXX`, `print(`, `console.`, `debugger` — none found.
- Verified the witness analysis: `all_ok * (o1 + o2 + o3)` is collapse-resistant within Phase-0 scope. The Phase-0 IR identity-lowers all wrapper constructors/eliminators, so the witness directly tests whether the i32 value survives 4 wrap + 4 unwrap operations + 1 causal transition. Any codegen bug introducing per-layer value drift would collapse one or more `o_i_ok` to 0; any typecheck regression would prevent compilation. The "vacuously passes if propagate returned identity" concern is by design — `propagate` IS specified as identity in Phase-0 (only the type-level kind changes from Cause to Effect), so identity behavior is the correct expected behavior.
- Confirmed no `helixc/tests/test_stage42_*` file exists. Per audit prompt note: Stage 42 added no new typecheck logic, so this matches the prior-stages pattern where SLIM stages rely on dogfood-as-only-test. Acceptable per Stage 42 progress doc's "no compiler-side changes" framing.
- Reviewed the three Stage 41 retrospective audit docs (`audit-stage41-inc4-gate1-{codereview,silent-failures,type-design}.md`): informational, document the silent-revert incident handled by hotfix `1e58862`. No Stage 42 scope impact.
- run.py "expects" string at line 110 references the initial 3-deep stack `Known<Present<WorldFrame<i32>>>` rather than the deepest 4-deep stack `Known<Present<WorldFrame<Cause<i32>>>>` — but this is a stylistic choice (showing initial state then "..."), not an error. Not flagged.
