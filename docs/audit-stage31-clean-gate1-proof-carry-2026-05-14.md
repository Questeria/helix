# Stage 31 Clean Gate 1 - Proof Carry And Duplicate Names

Result: CLEAN

Scope:
- `helixc/frontend/typecheck.py`
- `helixc/tests/test_typecheck.py`
- `helixc/tests/test_cli.py`
- `scripts/stage31_validate.py` quick proof list

Checks performed:
- Reviewed the current diff for structural refinement proof-carry.
- Confirmed proof reuse is exact and structural, not formatted-text based.
- Confirmed unsupported predicate shapes return `None` from structural keying and cannot prove by fallback class name.
- Confirmed generic-qualified names are rejected by predicate validation and constant evaluation.
- Confirmed duplicate type namespace names and duplicate constants fail closed before stale alias/constant data can be used.

Validation evidence:
- `python -m pytest -q helixc\tests\test_typecheck.py::test_stage31_duplicate_refinement_names_fail_closed helixc\tests\test_typecheck.py::test_stage31_equivalent_refinement_aliases_carry_exact_proofs helixc\tests\test_typecheck.py::test_stage31_unsupported_refinement_predicates_do_not_carry_by_name helixc\tests\test_typecheck.py::test_stage31_generic_qualified_refinement_names_are_unsupported helixc\tests\test_cli.py::test_stage31_emit_proof_obligations_json_for_equivalent_refinement_alias helixc\tests\test_cli.py::test_stage31_emit_proof_obligations_rejects_generic_refinement_name helixc\tests\test_cli.py::test_stage31_emit_proof_obligations_rejects_duplicate_proof_names helixc\tests\test_stage31_validate.py`
  - Result: `14 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: `stage31-quick: rc=0`

Findings:
- No blocking findings.

Residual risk:
- This intentionally remains syntactic proof reuse. Algebraic implication such as `self >= 0.0` proving `0.0 <= self` is still future SMT work and is tested as not carried today.
