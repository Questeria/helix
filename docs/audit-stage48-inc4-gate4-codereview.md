# Stage 48 Inc 4 Gate-4 Code-Review Audit

Gate-4 verification pass on the gate-3 patch (commit 3415727).
Read-only.

## Surface reviewed

- `helixc/frontend/typecheck.py`:
  - `_result_constructor_provenance` declaration + stewardship
    block (lines 589-621)
  - `_result_let_block_scopes` declaration (lines 622-628)
  - `check()` clear sites (line 677)
  - `_check_fn` clear (lines 2267-2274)
  - `_check_block` snapshot + scope-aware restore (lines
    2371-2462)
  - `_check_stmt` Let-arm prov + let-scope record (lines
    2464-2585)
  - `__try` call-dispatch arm (lines 4516-4661)
  - Assign-arm prov update (lines 5021-5035)
- `helixc/ir/lower_ast.py`:
  - `_lower_type` Result identity arm + Stage-49 TODO (lines
    843-882)
  - identity-lowered call tuple + `__try` entry + Stage-49 TODO
    (lines 2070-2103)
- `helixc/tests/test_stage48_try.py` (18 tests, full file).
- `docs/stage48-progress-2026-05-17.md` (full file).
- `docs/stage49-plan-2026-05-17.md` (full file).

## CRITICAL (90-100)

None.

## HIGH (80-89)

None. No CRITICAL or HIGH code-review findings — the gate is
not blocked.

## MEDIUM (filtered, confidence >= 80)

**CR4-M1 — Stale doc-comment in `__try` arm contradicts the
post-fix behaviour** (file `helixc/frontend/typecheck.py`,
lines 4534-4542, conf 92).

The comment block enumerating the `__try` validation rules
contains item 5:

> 5. Constructor-provenance: `Ok(7)?` is a degenerate
>    identity; `Err(7)?` is an unconditional early-return.
>    Neither is a Phase-0 bug per se (no runtime tag → both
>    fall through to the Ok-extract path identically). **We
>    allow both without a diagnostic here**; a dedicated lint
>    can flag them in a follow-up stage if dogfooding surfaces
>    them as real user mistakes.

This text is the pre-gate-1-F2 disposition. The actual code at
lines 4628-4646 REJECTS `err`-provenance name operands with a
dedicated `?` diagnostic ("`?` applied to `r`, which was
constructed via Err() — Phase-0 has no runtime Ok/Err tag
yet..."). A future reader of the doc-comment will believe the
arm is permissive when it isn't, and the gate-1 F2 regression
test will fail mysteriously when someone "restores" the
documented behaviour by deleting the rejection code.

**Suggested fix**: replace item 5 with the post-fix wording,
e.g.:

> 5. Constructor-provenance: `Ok(7)?` is benign (identity);
>    `Err(7)?` REJECTED with the gate-1 F2 diagnostic when the
>    operand is an A.Name with known "err" provenance (see
>    block at lines 4628-4646). The non-Name and dynamic-Err
>    cases remain F1 deferred (Stage 49 runtime tag fixes the
>    whole class).

**CR4-M2 — Cross-file line refs in `docs/stage49-plan-2026-05-17.md`
are stale** (file `docs/stage49-plan-2026-05-17.md`, lines 13-16,
conf 88).

```
- `is_ok(r)` / `is_err(r)` — currently rejected (Stage 46 closure
  gate-1 F1, see typecheck.py:4573-4632) because no tag exists.
- `map_err(r, new_err)` — currently rejected (Stage 46 F2,
  typecheck.py:4654-4694) for the same reason.
```

The actual locations post-gate-3 are:

- `is_ok` / `is_err` arm: `typecheck.py:4662` (start, runs to
  ~4730).
- `map_err` arm: starts at `typecheck.py:4737`-ish, well past
  4694.

The cited ranges 4573-4632 / 4654-4694 actually fall INSIDE
the new Stage 48 `__try` arm (4516-4661) — exactly the
opposite of what the document claims. Was wrong already at
gate-2 commit (verified with `git show c32dfbb:`); not a
gate-3 regression, but the Stage 48 closure is the natural
moment to either correct or convert to anchor-based references
(e.g., `# the `is_ok / is_err` arm in typecheck.py — search
for `if bn in ("is_ok", "is_err")`) that don't drift.

**Suggested fix**: replace numeric line refs with a stable
anchor token (search string) OR re-measure now. The
search-string approach prevents the next code edit from
re-introducing the drift.

**CR4-M3 — `_result_let_block_scopes` lacks the `check()` and
`_check_fn` clear sites its sibling has** (file
`helixc/frontend/typecheck.py`, lines 628 + 677 + 2274 +
599-611, conf 82).

The new stack `_result_let_block_scopes` (declared line 628) is
push/popped only inside `_check_block` (try/finally), and is
not reset in either `check()` (line 677 area) or `_check_fn`
(line 2274 area). Its sibling `_result_constructor_provenance`
IS cleared in both places (added in gate-2 M5 / Stage 46). The
pre-existing scope-stack pattern (`_local_const_scalar_scopes`)
also gets explicit `check()` reset at lines 680-681. The
stewardship comment block at 599-611 explicitly enumerates 6
sites for the dict but doesn't list any cleanup sites for the
parallel set-stack.

In normal operation push/pop balance inside a single check()
invocation, so this is not a live miscompile. But:
1. If a test or LSP harness reuses a TypeChecker instance and
   an exception escapes `_check_block` outside its outer try
   (theoretically possible during code-evolution), stale stack
   entries leak across `check()` invocations.
2. The stewardship-block omission undermines the gate-3 T-M3
   "centralizing helper" follow-up — a future maintainer adding
   a 3rd parallel stack will not see the precedent for clearing
   sites.

**Suggested fix**: add `self._result_let_block_scopes = []`
right after line 677 (next to the prov clear) and after line
2274 (next to the prov clear), AND extend the stewardship
comment at 599-611 with a parallel set-stack subsection listing
the same 6 sites.

## LOW (informational, conf 80-85)

**CR4-L1 — Cross-file `lower_ast.py:866` / `lower_ast.py:2097`
line refs in the F5 test comment are exact-line-fragile** (file
`helixc/tests/test_stage48_try.py`, lines 410-411, conf 82).

```
# Mirror the TODO(stage49) markers already in lower_ast.py:866 and
# lower_ast.py:2097.
```

Both line numbers happen to be accurate today, but any future
edit to either file (inserting comment lines, refactoring the
type-position vs expression-position arms) silently shifts
them. The Stage-49 maintainer is the most likely person to
hit this — exactly the audience the comment is written for.
Stage 49 is also when the markers will be CONSUMED and removed,
so the maintainer will be combing for them.

**Suggested fix**: switch to anchor-based reference, e.g.
`# Mirror the TODO(stage49) markers in lower_ast.py — search for
"TODO(stage49)" in `_lower_type` and the identity-lowered call
tuple.` The grep is O(1) and survives refactors.

**CR4-L2 — Order-warning banner for M5 test is positioned between
the F5 test and the G3-F1 tests, not next to the M5 test** (file
`helixc/tests/test_stage48_try.py`, lines 424-428 vs M5 test at
lines 342-373, conf 80).

```
# Order-sensitive note on M5 test above (per gate-3 code-review M3):
# the cross-fn carry test requires `maker` to be checked BEFORE
# `taker` so the stale provenance from maker pollutes taker's
# parameter check. The source order is intentional; do not
# reorder fn declarations.
```

A developer editing the M5 test (lines 342-373) won't see this
banner — the F5 test (lines 376-416) sits between them. The
banner is also overbroad post-fix: `_check_fn` now clears the
prov map at entry regardless of fn declaration order, so the
test passes whether or not `maker` is declared first. The
banner served gate-3 CR-M3's intent (warn future maintainers)
but its placement renders it nearly invisible to its audience.

Additional nuance: the banner says "do not reorder fn
declarations" but the actual order-sensitivity (if it existed)
would be about the order Helix source declares fns in the
TEST'S string source, not about the test functions themselves.
The wording is ambiguous.

**Suggested fix**: move the comment to immediately above the
M5 test docstring (or fold it into the docstring) AND clarify
that it's about the Helix-source `fn maker` / `fn taker`
declaration order, not the pytest function order. Or, if the
post-fix code is genuinely order-insensitive (which it is),
delete the banner and note in the test docstring that the
order is preserved for historical regression reproducibility.

## VERDICT

**CLEAN**.

- 0 CRITICAL, 0 HIGH findings → gate is not blocked.
- 3 MEDIUM findings are all documentation / stewardship polish
  with no live-correctness impact (CR4-M1 stale comment, CR4-M2
  stale line refs in stage49-plan, CR4-M3 missing cleanup
  symmetry for the new set-stack).
- 2 LOW findings are non-blocking polish.

Stage 48 Inc 4 closes. Recommend the MEDIUM fixes land at the
Stage 49 opening commit (CR4-M3 in particular dovetails with
the gate-3 T-M3 "centralizing helper" follow-up, and CR4-M2's
stage49-plan refs naturally update during Stage 49 work). CR4-M1
is the strongest candidate for a same-cycle inline fix since the
contradiction is in the file actively being modified — but
either timing is acceptable for code-review-class polish.

## Pattern observation

Gate-4 returns CLEAN (no escalating defect class) after gate-3's
G3-F1 scope-aware fix. The Stage 48 closure cascade matches the
Stage 46 4-gate rhythm: each silent-failure gate found a new
HIGH (gate-1 F2, gate-2 F1, gate-2 M5, gate-3 G3-F1); the
code-review verification gate (gate-4 here) catches only doc /
stewardship drift, not new live defects. This converges the
audit signal: the provenance-tracking design is now sound;
remaining work is documentation hygiene + the Stage 49 runtime
tag that eliminates the entire Phase-0 defect class.
