# Stage 49 Inc 5 Gate-4 Code-Review VERIFICATION Audit

Verification pass on the Stage 49 closure gate-1+2+3 fix sweep
(commit range `47d8f66..b4c8434`, 5 commits — db26e1c gate-1
Inc 1.5; c530891 gate-1+2 sweep; a5fb47f probe scrub; 8382bbf
f32 reject + parallel-agent polish; b4c8434 gate-3 G3-H1
wider-payload map_*).

Read-only. Verifies the gate-1 4 MEDIUMs closed AND scans for
new code-quality issues introduced by the sweep.

## Surface re-reviewed

- `helixc/frontend/typecheck.py` — Result-arm comments,
  `_reject_non_i32_result_payload` diagnostic, TODO(stage49)
  markers.
- `helixc/ir/lower_ast.py` — Result-position type arm, `__try`
  Inc 4 conditional branch, identity-tuple back-reference
  comments, Inc 1 placeholder comment.
- `helixc/backend/x86_64.py` — RESULT_PACK / RESULT_TAG /
  RESULT_PAYLOAD ops (lines 2186-2231).
- `helixc/tests/test_stage48_try.py` — module docstring +
  TODO(stage49) markers.
- `docs/stage49-plan-2026-05-17.md` — Tracker section.
- `docs/ROADMAP.md` — Stage 49 entry.
- `docs/stage48-progress-2026-05-17.md` — Out-of-scope section.
- Working-tree scratch-file scan.

## Gate-1 MEDIUM closure status

| ID | Finding | Status |
|----|---------|--------|
| CR1-M1 | Stale `_resolve_type` Result-arm comment | **FIXED** — typecheck.py:1420-1435 now says "Stage 49 Inc 1 LIFT" with explicit cross-references to `_lower_type` short-circuit, the G2-H1 payload-width reject, and the renamed pin test. |
| CR1-M2 | stage49-plan tracker unchecked | **FIXED** — all Inc 1-4 + Inc 1.5 marked `[x]` with commit SHAs (a08f21a / 2c3253c / 0868eae / 47d8f66 / db26e1c). Self-host + dogfood invariant rows also `[x]`. Inc 5 row carries detailed in-flight notes. |
| CR1-M3 | ROADMAP `(proposed)` | **FIXED** — ROADMAP.md:98 now reads `(in flight — Inc 1-4 + Inc 1.5 SHIPPED, Inc 5 closure audits in progress 2026-05-17)`. Lines 99-109 past-tense the unlocked builtins. |
| CR1-M4 | stage48-progress "Out of scope" stale | **FIXED** — stage48-progress-2026-05-17.md:361-369 prefixed with "Stage 49 update (2026-05-17): ..." + the two SHIPPED bullets struck through with commit SHAs. Lines 340-357 reframed as "LIFTED — see stage49 plan". |

All 4 gate-1 MEDIUMs are closed cleanly.

## CRITICAL (90-100)

None.

## HIGH (80-89)

None.

## MEDIUM (filtered, confidence >= 80)

**CR4-M1 — Stale `__try` Inc-1-placeholder comment block left
behind after Inc 4 made `__try` a real conditional branch**
(file `helixc/ir/lower_ast.py`, lines 2042-2047, conf 92).

The Inc 1 comment block at the top of the Result-constructors
section still reads:

> `__try(r)     -> RESULT_PAYLOAD(r)  (Inc 1 placeholder;
>                  the real conditional-branch IR ships in
>                  Inc 4 — for now __try extracts the Ok
>                  inner same as unwrap_ok, ...)`

Inc 4 (commit `47d8f66`) replaced this placeholder with a real
COND_BR + RETURN + payload-extract triple at lines 2171-2216
(the `expr.callee.name == "__try"` arm). A reader following the
flow from line 2042 sees a false claim about `__try` behavior
that is contradicted by code 130 lines below it.

**Suggested fix**: rewrite the `__try` line in the convention
block to `__try(r) -> COND_BR on RESULT_TAG; Err arm RETURNs
the packed Result, Ok arm extracts RESULT_PAYLOAD (Stage 49
Inc 4; see the dedicated arm at line 2171)`.

**CR4-M2 — Identity-tuple back-reference comment + obsolete
TODO(stage49) marker both claim `__try` still lives in the
identity-lowered tuple** (file `helixc/ir/lower_ast.py`, lines
2083-2099, conf 90).

Inside the identity-call tuple at lines 2237-2303, the comment
block at 2083-2095 (Stage 48 Inc 3 narrative) still says:

> Stage 48 Inc 3 — `?` propagation (`__try`) joins the
> identity-lowered set in Phase-0. With no runtime tag yet,
> every Result is shape-Ok ... `__try(r)` is observationally
> identical to `unwrap_ok(r)`. Stage 49 will add a real Ok/Err
> tag plus the conditional branch IR that gives `?` its actual
> propagation semantics — this lowering rule will then become
> the FALLBACK taken only when the operand is known at compile
> time to be Ok-shape.

Followed immediately by `# TODO(stage49): __try splits out of
this tuple when the runtime Ok/Err tag lands; it then becomes
a real conditional-branch` at line 2097. Both claims are
obsolete: `__try` is no longer in the identity tuple (verified
by reading the tuple contents at lines 2237-2301 — only the
wrapper-quintet, the cross-frame/temporal/modal/causal
transitions, and `"Ok", "Err", "unwrap_ok", "unwrap_err"`
remain; `__try` is gone, handled by its dedicated arm at
2171). The "FALLBACK taken only when ... known at compile time
to be Ok-shape" prediction never came true — Inc 4 made the
conditional unconditional. The TODO(stage49) marker is the
exact "Stage 49 closes Stage 48 TODOs" residue the audit
mission flagged.

**Suggested fix**: replace the lines-2083-2099 block with a
one-line back-reference: `# __try was REMOVED from this tuple
in Stage 49 Inc 4 (commit 47d8f66); see the dedicated
conditional-branch arm at ~line 2171.` Delete the
TODO(stage49) marker entirely.

**CR4-M3 — `test_stage48_try.py` module docstring still
describes `__try` as identity-lowered Phase-0** (file
`helixc/tests/test_stage48_try.py`, lines 10-14, conf 88).

The module docstring reads:

> IR lowering: identity-lowered (Phase-0). Without a runtime
> Ok/Err tag, every Result is observationally Ok-shape, so
> `r?` reduces to extracting the Ok inner — semantically
> identical to `unwrap_ok(r)` until Stage 49 lands the runtime
> tag + real conditional-branch IR.

This docstring is what `pydoc` / a future reader sees first
when opening the test file. Stage 49 Inc 4 has shipped both the
runtime tag AND the conditional-branch IR — the docstring's
"until Stage 49 lands ..." clause is now historical, not
forward-looking. Same comment-drift pattern as CR4-M1/M2 but in
a test file (which is the canonical place a maintainer goes to
understand `?` semantics).

**Suggested fix**: change line 10 paragraph to "IR lowering
(historical): Phase-0 Stage 48 identity-lowered `__try` to
unwrap_ok-equivalent extraction. Stage 49 Inc 4 (commit
47d8f66) replaced this with a real COND_BR + RETURN + Ok
payload-extract triple; the F2 / F5 pin tests below have been
polarity-flipped accordingly. The remaining `assert errs == []`
F5 reproducer is kept as a regression anchor."

## LOW (informational, conf 80-85)

**CR4-L1 — No `docs/stage49-progress*.md` ledger exists** (file
`docs/`, conf 84).

`ls docs/stage49-progress*.md` returns no match. Every prior
multi-Inc stage (35, 36, 46, 48) ships a `stageN-progress-
YYYY-MM-DD.md` ledger as the narrative log of what happened
per Inc + per gate. Stage 49 has the plan doc (`stage49-plan-
2026-05-17.md`), the gate-1 audit docs (codereview / type-
design / silent-failures), but no progress ledger. The plan
tracker has absorbed some of that role via the detailed Inc 5
row, but the historical convention is a separate file.

**Suggested fix**: not blocking for closure. Recommend writing
`docs/stage49-progress-2026-05-17.md` at the Stage 49 CLOSED
commit, consolidating the per-Inc narratives + the 3-gate
audit cascade summary. The plan doc tracker can then point at
it for the long-form history.

**CR4-L2 — TODO(stage49) markers in `typecheck.py:638, 670,
680` reference Stage 50+ collapse, not Stage-49-internal work**
(file `helixc/frontend/typecheck.py`, conf 83).

The three `TODO(stage49)` markers all read variations of "...
collapses when site-4 prov dict is removed" / "the runtime
Ok/Err tag obsoletes most of the Phase-0 static-provenance
machinery." The Stage 49 Inc 1.5 + gate-2 narrative in the
surrounding paragraph (lines 644-662) explicitly documents
that the static-provenance dict is KEPT as defense-in-depth
diagnostic — the Stage 50+ collapse is a deliberately-deferred
follow-up. The marker name `TODO(stage49)` is slightly
misleading given Stage 49 already shipped without doing this
collapse.

**Suggested fix**: rename to `TODO(stage50)` to reflect the
actual target stage. Not blocking; the surrounding text is
clear.

**CR4-L3 — Wider-payload diagnostic UX is excellent**
(observation, not a finding; file `helixc/frontend/typecheck.
py`, lines 6420-6430, conf 95 on the quality assertion).

The G2-H1 reject says:
> `{side}() payload type {ty} is not supported by the Stage 49
> packed-i64 Result representation; only i32 payloads work
> today`

with hint:
> `Stage 50+ widens the payload representation; for now, use
> Result<i32, i32> or wrap a wider value in a small i32
> handle. See docs/stage49-plan-2026-05-17.md:164-171.`

Names the side (Ok / Err / map_ok new_value / map_err
new_value), names the offending type, suggests a workaround,
references the planning doc with line numbers. This is the
right shape for dogfooding — a user hitting it can act without
filing a question. No fix needed.

## Working-tree hygiene

`ls _review_probe*.py _audit_*.py` returns no matches. Probe
scratch files from gates 1/2 (`_audit_gate2.py` etc.) were
swept clean in commit `a5fb47f`. `git status --short` (per
session start) shows only the audit-doc additions in `docs/`.

## VERDICT

**3/3 GATE-4 CLEAN — Stage 49 ready to CLOSE.**

- 0 CRITICAL, 0 HIGH findings — gate is not blocked.
- 4 gate-1 MEDIUMs verified CLOSED (CR1-M1/M2/M3/M4 all fixed
  cleanly by the gate-1+2+3 sweep + the parallel-agent
  ROADMAP update).
- 3 new MEDIUM findings: all comment-drift introduced by the
  Inc 4 + Inc 1.5 landings — CR4-M1 (stale `__try` Inc-1-
  placeholder comment), CR4-M2 (identity-tuple back-reference
  + obsolete TODO(stage49) still claiming `__try` lives in the
  tuple), CR4-M3 (test_stage48_try.py module docstring
  describes pre-Stage-49 identity-lowering). All three are
  doc-only, all in code Stage 49 actively touched, all
  trivially fixable in one closure-polish commit.
- 3 LOW findings: missing progress ledger (deferred-polish,
  conventional), TODO(stage49) marker name should be
  TODO(stage50) (cosmetic), wider-payload diagnostic UX
  explicitly affirmed as good.

The doc-drift pattern is the same Stage 46-48 closures have
consistently surfaced at later gates: structural code is
sound; the residual residue is comments and documentation
around the actively-modified arms. Recommend the 3 CR4-Ms land
inline at the Stage 49 closure commit (same DIFF-LOCALITY
argument that justified the gate-1 CR1-M1 same-cycle fix).

The gate-3 G3-H1 fix at typecheck.py:4879-4891 is structurally
correct AND has UX-quality diagnostic text — closure
recommendation stands.

## Pattern observation

Gate-4 verification confirms the gate-1+2+3 sweep landed
cleanly on the structural axis and on the gate-1 explicit doc-
drift list. The 3 NEW MEDIUMs are the second-order residue:
when Inc 4 (real `__try`) shipped AFTER Inc 1 (placeholder),
the Inc 1 placeholder comments stayed. Same defect class as
the gate-1 CR1-M1 stale `_resolve_type` comment — which was
itself created when Inc 1 landed the lift but didn't update
the comment Stage 48 had written. The pattern: every Inc that
upgrades a placeholder-mode arm should sweep the
comment-block immediately above the changed arm AND the
comment-block immediately above any sibling arm that mentions
the placeholder semantics. A future Stage NN should consider
adding a pre-commit lint that greps for `Inc N placeholder` or
`Stage NN will` once Stage NN's PR lands.
