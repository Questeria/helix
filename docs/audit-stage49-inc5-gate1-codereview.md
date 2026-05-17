# Stage 49 Inc 5 Gate-1 Code-Review Audit

Gate-1 code-review pass on the Stage 49 Inc 1-4 burst (commit
range `7eaba56..47d8f66`, 4 commits).
Read-only.

## Surface reviewed

- `helixc/ir/tir.py`:
  - New `OpKind` entries `RESULT_PACK` / `RESULT_TAG` /
    `RESULT_PAYLOAD` + their convention comment block
    (Inc 1 commit `a08f21a`, lines 299-339 of the post-diff
    file).
- `helixc/ir/lower_ast.py`:
  - `_lower_type` Result-arm rewritten (lines 843-882) from
    `_lower_type(ty.args[0])` to `TIRScalar("i64")`.
  - New dedicated call-arms for `Ok` / `Err` / `unwrap_ok` /
    `unwrap_err` / `__try` / `is_ok` / `is_err` (lines 2030-2152)
    inserted ABOVE the wrapper-quintet identity tuple.
  - Identity-call tuple (lines ~2155-2215) updated: `Ok`/`Err`/
    `unwrap_ok`/`unwrap_err`/`__try` removed from the tuple,
    replaced by a back-reference comment.
  - `map_ok` / `map_err` arm (lines ~2233-2322) upgraded from
    a thread-through to a SELECT-on-tag with packed-i64
    RESULT_PACK.
- `helixc/backend/x86_64.py`:
  - New `Asm` helpers `shl_rax_imm8` / `shr_rax_imm8` /
    `mov_eax_eax` (lines 930-940).
  - New `FnCompiler.compile_op` arms for `RESULT_PACK` /
    `RESULT_TAG` / `RESULT_PAYLOAD` (lines 2192-2237).
- `helixc/frontend/typecheck.py` (Stage 49-attributable subset
  of the 381-line diff — most of the diff is carried-over
  Stage 48 gate-5 cascade from commit `edd0d6f`):
  - `_resolve_type` Result arm (lines 1395-1419) — comment-only
    change to acknowledge G4-H1 deferral.
  - `is_ok`/`is_err` arm reject lifted (commit `2c3253c`),
    returns `TyPrim("bool")`.
  - `map_err` arm reject lifted (commit `0868eae`), returns
    proper `TyResult(ok=arg_tys[0].ok_ty, err=arg_tys[1])`.
  - `__try` Err-provenance reject lifted (commit `47d8f66`),
    rejection block at typecheck.py:4752-4772 deleted.
- `helixc/tests/test_stage49_runtime_tag.py` (738 lines, **34
  test functions** — note the mission brief says "72 test
  cases"; the file at 47d8f66 contains exactly 34 `def test_`,
  none parameterized).
- `helixc/tests/test_stage48_try.py`:
  - `test_stage48_closure_gate1_f2_err_constructed_question_rejects`
    renamed to `_lifted_by_stage49_inc4` + body inverted to
    assert typecheck-clean + run.
  - `test_stage48_closure_gate5_g4h1_result_of_wrapper_in_fn_signature_raises_at_ir`
    body inverted to assert lowering succeeds (see L2 below).
- `docs/stage48-progress-2026-05-17.md` (diff confined to a
  new "Gate-4/5 follow-up cascade" section).
- `docs/stage49-plan-2026-05-17.md` (mostly the cross-file
  ref tightening per CR4-M2 from the Stage 48 gate-4 audit).
- `docs/ROADMAP.md` (unchanged in the audit range — flagged
  below as M3).

## CRITICAL (90-100)

None.

## HIGH (80-89)

None. No HIGH code-review findings — the gate is not blocked.

## MEDIUM (filtered, confidence >= 80)

**CR1-M1 — Stale `_resolve_type` Result-arm comment claims a
limitation that Stage 49 Inc 1 lifted in the same burst**
(file `helixc/frontend/typecheck.py`, lines 1400-1418, conf 93).

The block reads:

> Stage 48 closure gate-5 type-design G4-H1 acknowledgement
> (deferred, audit Option 3): `Result<Known<...>, E>` in a
> FUNCTION-RETURN type position raises NotImplementedError at
> IR lowering because _lower_type's Result-arm identity-
> recurses into the Stage 37-41 wrapper-quintet which has no
> type-position arm. ... TODO(stage49): runtime Ok/Err tag +
> wrapper type-position arms eliminate the asymmetry.

Stage 49 Inc 1 (commit `a08f21a`) rewrote the `_lower_type`
Result-arm at `lower_ast.py:843-882` to short-circuit to
`TIRScalar("i64")` WITHOUT recursing into the Ok/Err inner
types. The NotImplementedError described here can no longer be
raised at this code path — verified by the companion pin test
`test_stage48_closure_gate5_g4h1_result_of_wrapper_in_fn_signature_raises_at_ir`
which was updated in the same range to assert the LIFT (`assert
typecheck(prog) == []` + `assert "ret_known" in m.functions`).

The `TODO(stage49)` marker at the bottom of the comment is
exactly the "Stage 49 closes Stage 48 TODOs" question from the
audit mission — and the answer here is: the TODO is consumed
but the marker + the preceding paragraph still claim the
limitation is live.

**Suggested fix**: replace the block with a one-paragraph
"LIFTED by Stage 49 Inc 1: Result type-arm no longer recurses
into wrapper inner; see lower_ast.py Result-arm + the pin test
renamed in test_stage48_try.py" or delete the comment entirely
since the design is now uniform.

**CR1-M2 — `docs/stage49-plan-2026-05-17.md` Tracker checkboxes
all unchecked despite Inc 1-4 having shipped** (file
`docs/stage49-plan-2026-05-17.md` at commit 47d8f66, lines
~155-165 of that revision, conf 95).

At 47d8f66 the plan tracker reads:

```
- [ ] Inc 1: TIR pack/tag/payload + Ok/Err/unwrap_ok/unwrap_err
- [ ] Inc 2: is_ok / is_err
- [ ] Inc 3: map_err
- [ ] Inc 4: real `?` propagation branch
- [ ] Inc 5: 3-clean-gate closure
- [ ] Self-host cascade G2..G4 byte-identical after each Inc
- [ ] dogfood_17 still exits 42 throughout
```

All four shipped Increments have unchecked boxes; the
self-host and dogfood invariants are likewise unchecked
despite the test file containing `test_stage49_inc1_dogfood_17_still_exits_42`
and `test_stage49_inc4_dogfood_17_still_exits_42` that
empirically validate them. This is exactly the "Inc-by-Inc
tracker checkboxes" question from the audit mission. (The
working-tree file post-commit-`db26e1c` has the boxes flipped
to `[x]` — confirming the omission was caught by the gate-1
follow-up that's in flight per the user's Inc 5 prep.)

**Suggested fix**: mark Inc 1-4 + the two invariant rows as
`[x]` with commit SHAs (the post-`db26e1c` revision shows the
intended format). Also add an "Inc 1.5: runtime tag-check"
row since the audit lanes are already discussing it and it is
the natural follow-up.

**CR1-M3 — `docs/ROADMAP.md` still tags Stage 49 as
`(proposed)` despite 4 shipped Increments** (file
`docs/ROADMAP.md` at commit 47d8f66, line 98, conf 88).

```
- **Stage 49** (proposed): Tier 4 #14 Inc 3 — runtime Ok/Err
  tag. Unlocks the 4 currently-rejected builtins ...
```

ROADMAP has not been touched in the Stage 49 range. Even if
the convention is to flip to ✅ DONE only after Inc 5 closes,
the SHIPPED state of Inc 1-4 deserves an "in flight" marker
to distinguish from "(proposed)" picks that have not started.
The Tier 4 #14 Inc 3 status line elsewhere should also note
that `is_ok`/`is_err`/`map_err` are no longer rejected (this
was Stage 49's headline payoff).

**Suggested fix**: change `(proposed)` to `(in flight — Inc
1-4 SHIPPED, Inc 5 closure audits in progress 2026-05-17)`.
Update lines 99-100 to past-tense for the 3 unlocked builtins.
Stage 49 declaring CLOSED later will flip this to ✅ DONE in
one edit.

**CR1-M4 — `docs/stage48-progress-2026-05-17.md` "Out of scope
(Stage 49+)" section still lists items Stage 49 has shipped**
(file `docs/stage48-progress-2026-05-17.md`, lines ~355-365,
conf 86).

```
### Out of scope (Stage 49+)

- Real runtime Ok/Err tag (IR opcode for discriminated union).
- The runtime `?` early-return branch.
- ...
```

Both bullets are exactly what Stage 49 Inc 1 + Inc 4
delivered. Reading stage48-progress as the historical record
of Stage 48's scope is defensible, but the same document was
RE-EDITED in commit `edd0d6f` (after the initial close) to
add a "Gate-4/5 follow-up cascade" section — i.e., the doc is
not frozen-historical, it's still being maintained. A future
reader landing on "Out of scope" will believe these are open
Phase-1 items.

The mission asked specifically: "Does `docs/stage48-progress-
2026-05-17.md` correctly cross-reference Stage 49's completion
of its TODOs?" Answer: NO — no cross-reference added (and the
"Phase-0 vs Stage 49+ semantic upgrade" section at lines
339-353 is also written in future tense despite the work
having completed).

**Suggested fix**: add a brief "Stage 49 update (2026-05-17):
the first two bullets SHIPPED in Stage 49 Inc 1 + Inc 4; see
`docs/stage49-plan-2026-05-17.md`" prefix to the "Out of
scope" list, AND past-tense the "Once Stage 49 adds..."
paragraph at line 341.

## LOW (informational, conf 80-85)

**CR1-L1 — Test-file count mismatch with mission brief** (file
`helixc/tests/test_stage49_runtime_tag.py` at 47d8f66, conf 82).

The mission brief states "738 lines, 72 test cases" but
`grep -c "^def test_"` returns **34** test functions and there
are no `@pytest.mark.parametrize` decorators (`grep -c
parametrize` returns 0). The line count of 738 is correct; the
test-count figure of 72 is off by ~2x.

Not a bug in the code, but the audit's "is every Inc covered?"
question becomes load-bearing on a wrong premise. After
re-checking with the actual count: Inc 1 has 13 tests (opcode
registry, type lowering, IR shape for Ok/Err/unwrap/try, +
end-to-end exit codes for round-trip / call-return / Err-payload
/ dogfood-17 / large-payload / chained-Ok); Inc 2 has 7 (4
static + 1 dynamic + 2 negative); Inc 3 has 8 (4 happy-path on
map_ok/map_err × Ok/Err + 2 tag-preservation + 2 negative); Inc
4 has 6 (Ok-fall-through, Err-propagate, chained-3-level,
post-`?`-code-skip, `r? + 2` arithmetic, dogfood-17 regression).
**Coverage IS adequate per Increment.** The mission-stated
shortfall ("specifically: Inc 1: ... unwrap-wrong-arm halt")
about Inc 1.5 is correctly identified as a gap — but Inc 1.5
isn't in the 7eaba56..47d8f66 range; it's deferred per the
stage49-plan tracker and is the next commit (`db26e1c`).

**Suggested fix**: none. Note for the next audit cycle to
re-baseline test-count claims.

**CR1-L2 — `test_stage48_closure_gate5_g4h1_..._raises_at_ir`
test name is now misleading post-Stage-49 Inc 1** (file
`helixc/tests/test_stage48_try.py`, line 633, conf 84).

The test was renamed in spirit (the docstring + body were
flipped to assert the LIFT — typecheck clean + IR lowering
succeeds), but the function name still ends in `_raises_at_ir`.
A future maintainer skimming pytest output or grep results for
"raises" will get a wrong mental model.

**Suggested fix**: rename to
`test_stage48_closure_gate5_g4h1_result_of_wrapper_in_fn_signature_lowers_clean_post_stage49_inc1`
(matches the rename convention applied to the gate-1 F2 test).

**CR1-L3 — `RESULT_PAYLOAD` op result type hardcoded as
`TIRScalar("i32")` regardless of declared Result inner type**
(file `helixc/ir/lower_ast.py`, lines 2091 + 2131, conf 82).

`unwrap_ok` / `unwrap_err` / `__try` all emit:
```python
self.builder.emit(tir.OpKind.RESULT_PAYLOAD, packed,
                  result_ty=tir.TIRScalar("i32"))
```

For `Result<i32, i32>` this is exactly correct. For
`Result<i64, i32>` or `Result<bool, i32>`, the hardcoded i32
result-type would mis-classify the downstream slot. The Inc 1
plan acknowledges this in the convention block on `OpKind.
RESULT_PACK` ("both currently constrained to i32 for Inc 1"),
and the Phase-0 dogfood corpus only exercises i32/i32 — so
not a live bug. But there is also NO typecheck-side reject
preventing a user from declaring `Result<i64, i32>` and
hitting the silent truncation.

**Suggested fix**: either (a) add a typecheck-side reject for
non-i32 Result inner types with a "Stage 50+: wider payloads"
diagnostic, OR (b) inherit the result_ty from the operand's
TyResult.ok_ty / err_ty + raise a NotImplementedError at IR
lowering for non-i32 widths (matches the loud-fail pattern of
the wrapper-quintet `_lower_type` arm). Both align with the
"fail loud, never silently miscompile" doctrine from Stage 46
gate-1.

**CR1-L4 — `RESULT_PACK` opcode comment block correctly
documents the high-tag/low-payload convention, but the parallel
`x86_64.py` comment at lines 2192-2200 partially restates it;
keep them DRY** (files `helixc/ir/tir.py` lines 299-339 +
`helixc/backend/x86_64.py` lines 2192-2200, conf 80).

Both files independently spell out:
> packed = (tag << 32) | (payload & 0xFFFFFFFF)
> tag    = packed >> 32 (logical, zero-extending)
> payload = packed & 0xFFFFFFFF

The duplication is GOOD for the inline-doc-completeness Q7
("future reader doesn't reverse them"). The minor risk is the
two comment blocks drifting in a future edit. The x86_64
comment correctly says "mirrors the block on OpKind.
RESULT_PACK in tir.py" which acts as a back-reference; that's
the right discipline. Logging this purely so a future audit
catches drift if one of the two blocks is edited and the
other isn't.

**Suggested fix**: none for now. If a third site (e.g. a
debugger pretty-printer) is added, factor the convention into
a shared docstring constant.

## VERDICT

**CLEAN**.

- 0 CRITICAL, 0 HIGH findings — gate is not blocked.
- 4 MEDIUM findings: all documentation / stewardship drift
  (CR1-M1 stale typecheck Result-arm comment; CR1-M2 stage49-
  plan tracker unchecked; CR1-M3 ROADMAP still `(proposed)`;
  CR1-M4 stage48-progress "Out of scope" stale). The
  comment-drift pattern is the same one Stage 48 closures
  consistently surfaced at the code-review gate: structural
  correctness is sound; what slips is the documentation around
  it. Naming convention (`RESULT_PACK` / `RESULT_TAG` /
  `RESULT_PAYLOAD`) matches the existing TIR snake-case dotted
  style (`result.pack` etc.) per question Q6.
- 4 LOW findings: count mismatch in the mission brief
  (CR1-L1, audit-meta not code), misleading legacy test name
  (CR1-L2), hardcoded i32 payload-result-ty (CR1-L3 — known
  Phase-0 limit, worth a typecheck guard at Stage 50), DRY
  caveat on the convention comment (CR1-L4).

Stage 49 Inc 4 is structurally ready to close. Recommend the
MEDIUM fixes land at the Stage 49 closure commit (CR1-M2 and
CR1-M3 in particular block the "Stage 49 SHIPPED" signal that
ROADMAP + plan need to convey). CR1-M1 is the strongest
candidate for same-cycle inline fix since the contradiction is
in a file actively-modified by Stage 49 commits.

## Pattern observation

Gate-1 code-review returns CLEAN with the expected
documentation-drift residue. The Stage 49 Inc 1-4 burst is a
clean "replace identity-lowering with packed-tag" mechanical
transform; the typecheck arms that were rejecting now return
real types, the lowering arms that were threading-through now
emit real IR, the backend gets three new opcodes with correct
SysV width handling. The cascading-defect rhythm that Stage 46
and Stage 48 surfaced (gate-1 reveals nothing structural;
gate-2 surfaces a dynamic silent-failure) is the expected next
shape — gate-2 silent-failure should probe specifically the
no-runtime-tag-check on unwrap_ok/unwrap_err (the explicit Inc
1.5 deferral at `lower_ast.py:2085-2090`) as the candidate
silent-miscompile pattern most likely to surface.
