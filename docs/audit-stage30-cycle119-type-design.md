# Stage 30 Cycle 119 Type-Design Audit

Verdict: PASS

Cycle 119 tightened the type boundary between source, Tensor IR, Tile IR, and PTX. The type-design audit initially found contract mismatches, all fixed in this cycle:

- Mutable parameters now typecheck as mutable and lower through allocated mutable slots.
- Indexed assignment targets must be named array or tile bindings; non-place indexed assignments reject.
- GPU index builtins typecheck only in call form inside `@kernel`.
- `@kernel` PTX surface is explicit: HBM tile parameters must be 1D, use supported `f32` or `i32` element types, and return `()`.
- `@kernel extern` declarations go through the same HBM ABI validation, and PTX does not emit fake bodies for declaration-only kernels.
- PTX scalar and HBM direct Tile IR contracts are now checked for exact operand counts, result counts, value types, register classes, result types, attrs, and dtype agreement.
- Unsupported PTX scalar widths and unsupported HBM element types fail closed instead of pretending to be f32/i32.

Verification:

- `python -m pytest helixc\tests\test_typecheck.py -q` -> 133 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 63 passed
- `python -m pytest helixc\tests\test_ptx.py -q` -> 52 passed
- `python -m pytest helixc\tests\test_cli.py -q` -> 58 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- Focused codegen slice -> 38 passed, 667 deselected
- Bootstrap slice -> 18 passed, 687 deselected

Auditor result: PASS with high confidence. The final probes confirmed the source typechecker, lowering, direct PTX emitter, and CLI now agree on the supported Stage 30 PTX surface.
