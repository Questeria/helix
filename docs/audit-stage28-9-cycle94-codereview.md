# Audit Stage 28.9 cycle 94 — Code review

Scope: HEAD `85bece0` (cycle-93 fix-sweep landing F1 kind-coherence + C92-1 docstring rewrite).

Mode: STRICT READ-ONLY. No edits made. One Write only (this doc).

## Review targets (NARROW — cycle-93 quality only)

Prior C1–C93 findings and known-deferred items are NOT re-flagged. Parallel
Stage 28.10 / 28.11 work is INDEPENDENT and not in scope.

### 1. F1 regression-test docstrings — clear and discriminative?

`test_c92_f1_intlit_with_float_suffix_rejected` (test_typecheck.py:1742):
docstring names the exact bad shape (`42_f32`), the pre-fix pipeline path
(IntLit → TyPrim('f32') → CONST_INT(TIRScalar('f32')) → raw bit-pattern
0x2A in f32 slot), and the numeric consequence (5.88e-44 vs 42.0). Cites
F1 + HIGH conf 85. Discriminates from the symmetric and accept-case
siblings.

`test_c92_f1_floatlit_with_int_suffix_rejected`: short, names the bad
shape `4.2_i32`, explicitly cross-references as "symmetric". Adequate
given the longer first docstring.

`test_c92_f1_valid_intlit_int_suffix_accepted`: states the positive
shape `42_i32` and the no-diagnostic expectation. Discriminative.

Verdict: clear and discriminative.

### 2. Cycle-93 docstring rewrite on `test_stdlib_vec_first_legacy_api`

The rewritten docstring (test_codegen.py:11184) now correctly states the
legacy body uses `__arena_push(val)` + 2-arg `vec_first(arena_base, len)`,
and the canonical sibling uses 3-arg `vec_push(arena, idx, val)`. Body
inspection at test_codegen.py:11205–11208 matches; canonical sibling at
test_codegen.py:12825–12839 matches. The acknowledgment of the
cycle-91 mischaracterisation is included. The `test_stdlib_vec_last_legacy_api`
docstring is updated symmetrically and the legacy body at 11222–11227
matches.

The "near line 12814" / "near line 12832" callouts in the new docstrings
are off-by-11/off-by-10 (actual canonical defs are at 12825 / 12842).
"Near" hedges, so this is not a docstring lie; flagging as <75% confidence
and not counted.

Verdict: accurate.

### 3. `_FLOAT_PRIM_NAMES` / `_INT_PRIM_NAMES` location

Defined at module scope in `helixc/frontend/typecheck.py:379–383`, immediately
before `class TypeChecker`. Comment block at 374–378 names the audit/fix and
purpose. Frozensets are immutable, hashable, share across instances, and have
O(1) membership — appropriate for the per-IntLit/FloatLit check site. Naming
matches existing module-private convention (leading underscore). No state
dependency, no instance-method coupling. Module scope is correct.

Verdict: appropriate.

### 4. Other docstring lies / stale comments in recently-touched files

Checked the inline comments added at typecheck.py:1212–1228 and 1230–1241
against the actual fix logic and against the AST nodes (IntLit / FloatLit
`type_suffix` field semantics): the prose matches the code. Error-message
suggestion `42.0_f32` is syntactically a valid Helix float literal and
matches the rejected example. No drift between comment and code.

`_FLOAT_PRIM_NAMES` set covers `{f16, bf16, f32, f64}` — matches every
float prim recognised by lexer/parser suffix tables (confirmed by spot-check
that no other float-prim name appears in the recently touched files).
`_INT_PRIM_NAMES` covers signed/unsigned i8..i64 + isize/usize, matching
the canonical integer suffix set.

Verdict: no findings.

## Findings at conf ≥ 75%

None.

## Result

**PASS** — 0 findings at confidence ≥ 75%.
