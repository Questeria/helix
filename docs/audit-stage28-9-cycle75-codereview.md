# Audit Stage 28.9 cycle 75 — Code review

Scope: HEAD `9a51cbf` (cycle-74 fix-sweep addressing cycle-73 totality
double-descent CN-1). NARROW conservative code-review of cycle-66..74
deltas only — broader-pipeline deferred-known items (struct_mono
pre-flatten asymmetry, monomorphize _mangle_ty catchall, hash_cons
_ast_equal SHA-256 fallback, typecheck-before-flatten in check.py)
are NOT re-flagged per cycle-67/70/71/72 deferral discipline.

Audit boundary (focus areas):
1. Comment vs. code drift in `helixc/frontend/totality.py` post
   ASTVisitor migration + cycle-74 double-descent fix.
2. Missing regression tests for cycle-74 double-descent fix.
3. TODO/FIXME residue in cycle-66 `flatten_modules` intra-mod aliases
   code (including cycle-68 follow-on tightening).
4. Cycle-71 polish completeness — does test coverage match the
   surgical `StructLit.name` rewriting fix?

## Verdict

**PASS** — 0 findings at confidence >= 75%.

## Probe-by-probe summary

### Focus 1 — Comment/code drift in totality.py

- Module docstring (lines 1-25) describes "Conservative structural
  recursion checker" and still matches the actual algorithm
  (subtract / divide by positive const, conservative reject
  otherwise).
- Migration comment (lines 35-50) accurately describes the cycle-71
  hand-rolled `_children` -> `ASTVisitor` migration AND the cycle-58
  parity with panic/unsafe/trace/deprecated passes.
- `_SelfCallCollector` class docstring (lines 54-58) is accurate:
  ASTVisitor's `generic_visit` walks every dataclass field —
  consistent with `ast_walker.py:199-211`.
- `visit_Call` cycle-73-fix inline comment (lines 68-76) precisely
  documents the auto-descent contract from `ast_walker.py:191-196`
  and cross-references the sister `panic_pass._PanicCollector`
  pattern, which I verified at `panic_pass.py:65-72` (panic also
  omits an explicit `generic_visit` call, relying on the same
  auto-descent).
- `check_totality` docstring (lines 80-91) accurately notes the
  cycle-58 / cycle-71 evolution to `iter_fn_decls` discipline.
- No stale references to the removed `collect_items` or `_children`
  helpers.

Result: no drift at >= 75% conf.

### Focus 2 — Cycle-74 regression test sufficiency

`test_c73_cn1_totality_no_double_descent` in `helixc/tests/test_deprecated.py`
(line 608+) directly exercises the defect: a recursive function whose
body is `rec(rec(n - 1))`. Pre-fix the double-descent would have
counted the inner self-call twice (4+ entries in `collector.calls`);
post-fix the assertion locks in exactly 2 (outer + inner). The test
instantiates `_SelfCallCollector` directly and calls `.visit(fn.body)`,
isolating the visitor-contract behavior from broader pipeline
interactions. Coverage matches the size/shape of the surgical fix.

Stylistic note (sub-75): the test is placed in `test_deprecated.py`
rather than `test_totality.py`. This mirrors the cycle-71
`test_c71_struct_lit_*` placement convention (also in test_deprecated.py
despite being a flatten_modules concern), so it is precedent-consistent.
Below the 75% threshold for raising as a finding.

Result: regression test sufficient. No finding at >= 75% conf.

### Focus 3 — TODO/FIXME residue in cycle-66 intra-mod aliases code

`grep -nE 'TODO|FIXME|XXX|HACK' helixc/frontend/flatten_modules.py` =>
no matches. The cycle-66 / cycle-68 intra-mod aliases logic
(`flatten_modules.py:86-229`) is documented with explicit cycle
provenance and ROOT-CAUSE / FIX prose; no deferred markers leaked
into source.

The cycle-66 / cycle-68 evolution comments correctly describe the
single-index `direct_lifts_start` -> per-direct-lift
`local_lift_indices` migration and the type filter narrowing
(`(A.FnDecl, A.StructDecl, A.EnumDecl, A.ConstDecl, A.TypeAlias)`)
that excluded the bogus ModBlock / AgentDecl alias capture.

Result: no finding at >= 75% conf.

### Focus 4 — Cycle-71 polish completeness vs. StructLit.name fix

The cycle-71 surgical fix is a single line in
`flatten_modules._rewrite_expr`'s StructLit arm:
`new_name = aliases.get(e.name, e.name)`. The cycle-71 polish commit
(`27945ed`) adds two tests:
- `test_c71_struct_lit_name_mangled_in_mod` — positive case (mod
  `m { struct Foo; fn make() { Foo {x:1} } }` -> StructLit.name
  `m__Foo`).
- `test_c71_struct_lit_top_level_unchanged` — negative case
  (top-level StructLit not rewritten).

The fix has one alias-mapping site and one fallback branch. The two
tests pin both branches via `assert struct_lit.name == "m__Foo"`
and `assert struct_lit.name == "Bar"`. The cycle-72 audit-clean
commit (`a22cba0`) claims "coverage of all 4 StructLit positions"
— I read this as referring to the audit's invariant probes
(positive remap, negative no-rewrite, plus the StructLit arm
visited via Block.final_expr / Let.value / Call.args / nested
contexts), not a literal "4 tests" claim. The two tests are
sufficient for the surgical fix's branch coverage.

Sub-75 observation (NOT raised as finding): the tests both probe
StructLit at `Block.final_expr` position. They do not directly
probe StructLit inside `Let` RHS, `Call.args`, or nested `If`
branches. However, `_rewrite_expr` recursion through Block/Let/If/Call
is already exercised by other cycle-65..68 tests
(`test_c65_cn1_intra_mod_calls_rewritten`, etc.), so adding
positional StructLit variants would be duplicate coverage of the
walker recursion, not the fix's alias-lookup line. Below 75% conf
threshold.

Result: test coverage matches fix surface. No finding at >= 75% conf.

## No source files were edited.

This audit is strict read-only per instruction. The only file
written is this audit doc.

## Targeted suite gates (verification)

- `python -m pytest helixc/tests/test_deprecated.py -q` => 28 passed.
- `python -m pytest helixc/tests/test_totality.py -q` => 15 passed.

Both targeted suites are green at HEAD `9a51cbf`, confirming the
cycle-74 fix has not regressed downstream coverage.

## Counter

Cycle 75 = clean (0 findings). Per cycle-72 "2/5" counter, this
extends the streak. The narrow-scope strategy continues to converge
as designed.
