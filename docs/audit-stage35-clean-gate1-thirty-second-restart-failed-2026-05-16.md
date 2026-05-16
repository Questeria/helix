# Stage 35 Clean Gate 1 - Thirty-Second Restart Failed

Date: 2026-05-16
Base HEAD: `fb9400d` (`Fix Stage 35 thirty-first restart findings`)
Result: NOT CLEAN. Stage 35 remains at `0/3` clean gates.

Restart 32 began from the pushed restart-31 fix sweep. Support baseline was
green, then three fresh audit lanes found remaining runtime, CLI, and
continuation-document blockers. The blockers were fixed in this restart, but
because the gate found issues, this restart does not count as a clean gate.

## Audit Findings

### Lane A - AD, Tensor, Runtime, NN

Status: findings present.

- `tf2d_row_sum`, `tf2d_col_sum`, and `tf2d_diag` could write through
  undersized destination buffers and still return success.
- `rev_count` and `rev_cap` trusted fake tape handles and exposed forged
  metadata values.
- Working-memory tick corruption was accepted as valid state.
- Working-memory and episodic-memory tick increments could overflow and either
  corrupt or poison otherwise valid objects.

### Lane B - Backend, PTX, CLI

Status: findings present.

- Direct x86 accepted a flag-shaped output filename such as `--no-stdlib` and
  wrote a binary there.
- `helixc.check -o` accepted a following flag as an output path.
- `--emit-ast` mixed banner and parse progress diagnostics into stdout before
  the AST artifact.
- stdout emit modes such as `--emit-ir` silently ignored `-o` and exited
  successfully without writing the requested file.

### Lane C - Docs and Public Claims

Status: findings present.

- The restart-31 audit doc and Stage 35 progress ledger still said to commit
  restart 31 and begin restart 32 after restart 31 had already been committed
  as `fb9400d`.

## Fix Sweep

- 2D f32 output helpers now validate destination slices before writing.
- Reverse-AD metadata accessors now call `rev_tape_valid` and fail closed on
  forged handles.
- Working-memory validation now rejects negative ticks, and working/episodic
  mutators reject max-int ticks before incrementing.
- `helixc.check -o` now rejects flag-shaped output values.
- Direct x86 now rejects flag-shaped output paths before compiling.
- stdout emit modes now reject `-o` instead of silently ignoring it.
- `--emit-ast` now keeps diagnostics on stderr so stdout is artifact-only.
- Continuation docs now record restart 31 as committed/pushed and restart 32 as
  the current failed fix sweep.

## Verification

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf2d_output_helpers_reject_short_destinations or revad_metadata_accessors_reject_fake_tape or agi_memory_rejects_corrupt_and_overflow_ticks"`
  - Result: 3 passed, 884 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "parse_args_output_rejects_flag_value or emit_ir_with_output_is_error or output_flag_value_rejected_without_writing or direct_x86_rejects_flag_shaped_output or main_emit_ast"`
  - Result: 5 passed, 185 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf2d or t2d or tensor or revad or agi_memory"`
  - Result: 78 passed, 809 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86"`
  - Result: 65 passed, 125 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 40 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 34 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 45 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 24 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,354 tests collected.
- `git diff --check`
  - Result: passed.

## Next Step

Commit and push this restart-32 fix sweep, then begin restart 33 as another
fresh Stage 35 clean-gate attempt from the newest pushed HEAD.
