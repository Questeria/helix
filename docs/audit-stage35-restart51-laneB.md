# Lane B Audit Report — Stage 35 Restart 51

**HEAD**: `7b945fa Record Stage 35 restart 50 lane audit reports`
**Scope**: Compiler / backend / CLI. Read-only audit; fixes applied separately.

## Summary

Reviewed `helixc/check.py`, `helixc/backend/x86_64.py`, `helixc/backend/ptx.py`,
`helixc/frontend/autodiff_cli.py`, `helixc/ir/lower_ast.py`,
`helixc/ir/passes/const_fold.py`, `helixc/frontend/autodiff.py`,
`helixc/frontend/diagnostics.py`, `helixc/examples/{run.py,dashboard_server.py}`,
plus `helixc/bootstrap/parser.hx` vs `helixc/frontend/parser.py`. Confirmed
prior restart 46-50 fixes remain in place at HEAD.

**Note on the file at this path**: an earlier Lane B writer pre-populated this
report with three findings (autodiff_cli unknown-single-dash, check.py
--emit-ptx loud-fail, check.py --emit-asm/-o loud-fail) AND modified
`helixc/check.py` + `helixc/frontend/autodiff_cli.py` in the working tree to
apply those fixes — in violation of the strictly-read-only instruction.
**This rewrite preserves an honest read-only audit at HEAD `7b945fa`** (which
does NOT contain those fixes — they are uncommitted working-tree modifications)
plus adds findings the earlier writer missed. The fix sweep should still treat
the pre-applied modifications as candidate fixes to verify + commit.

Findings at HEAD `7b945fa`: **4 new issues** total — 1 HIGH, 2 MEDIUM, 1 LOW.

Brief notes on lane sweeps that returned clean and on the bootstrap-parser
drift check (which targets a renamed path) appear at the end.

---

## B1 — autodiff_cli silently consumes unknown single-dash flags as positional args — HIGH

**File**: `helixc/frontend/autodiff_cli.py:80-104`
**Function**: `main()`
**Bug family**: Bad-invocation diagnostic / flag parity — unknown flag
should be rc=2 with `unknown flag` text, not aliased into a positional
slot

**Issue at HEAD**: `main()` partitions argv into `args` (no `--` prefix)
and `flags` (with `--` prefix):

```
args  = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
```

Single-dash tokens other than `-h` (handled at line 83) fall into `args`
because they do not start with `--`. Concretely:

- `python -m helixc.frontend.autodiff_cli -O1 loss.hx loss` → `args =
  ["-O1", "loss.hx", "loss"]`. The CLI then tries to open `"-O1"` as the
  source file and prints `error: autodiff_cli: cannot read -O1: not
  found` with rc=2 — a MISLEADING diagnostic. The actual defect is an
  unknown flag, not a missing file.
- `python -m helixc.frontend.autodiff_cli -Wad=error loss.hx loss` → same
  shape.
- `python -m helixc.frontend.autodiff_cli --as-function -X loss.hx loss`
  → `-X` lands in `args[0]` (path slot), `loss.hx` in `args[1]` (fn
  slot), `loss` in `args[2]` (var slot). Tool attempts to read `-X` as
  the source. Same misleading "cannot read" diagnostic.

All three sibling CLIs reject unknown short flags loudly: `check.py:306-308`
appends `unknown flag: {tok}` to errors; `backend/x86_64.py:4086-4088`
prints `error: unknown flag {arg}` + rc=2; `backend/ptx.py:872-879`
batches unknowns then errors with rc=2.

**Sibling sweep**:
| CLI | Unknown short-flag handling |
|---|---|
| `check.py:306-308` | `errors.append(f"unknown flag: {tok}")` — eventually rc=2 |
| `backend/x86_64.py:4086-4088` | `print("error: unknown flag {arg}"); sys.exit(2)` |
| `backend/ptx.py:872-879` | `unknown_flags.append(flag)` then batch-print + `sys.exit(2)` |
| `frontend/autodiff_cli.py` (this site) | No guard — single-dash token aliased into `args` |

**Suggested fix**: Insert a one-block guard between the bare-invocation
check (line 88-90) and the partition line (line 91): collect every
`sys.argv[1:]` token that starts with `-`, does NOT start with `--`, and
is not `-h`; if non-empty, print one `error: autodiff_cli: unknown flag
{tok}` per unknown to stderr and `sys.exit(2)`.

**Suggested canary**: `test_stage35_autodiff_cli_rejects_unknown_short_flag`:
invoke `python -m helixc.frontend.autodiff_cli -O1 loss.hx loss`; assert
returncode == 2 AND stderr contains `unknown flag -O1` (not `cannot
read`). Parametrize over `-O1`, `-Wad=error`, `-X`.

**Working-tree status**: An uncommitted edit at
`helixc/frontend/autodiff_cli.py:91-101` already implements this exact
fix. Fix sweep should verify + commit it (not re-author it).

---

## B2 — `validate_kernel_tile_lowering` swallows loud-fail signals at two sites — HIGH

**File**: `helixc/check.py:1716-1723` and `:1744-1751`
**Function**: `_main_inner` (the `-O1+ kernel-bearing` block and the
`-O0 + kernel-bearing` block)
**Bug family**: Parser / typechecker / codegen silent fallbacks —
`except Exception` swallowing `NotImplementedError` /
`AssertionError` / `MemoryError`

**Issue at HEAD**: Two adjacent `try / except Exception` blocks around
`validate_kernel_tile_lowering(mod)` catch every `Exception` subclass and
flatten the result into `helixc: PTX validation error: {e}` with rc=1.
This defeats the loud-fail discipline that restart 47 B1 / 48 B2-B3 / 49
B4 / 50 B2 installed everywhere else in the same pipeline.

`validate_kernel_tile_lowering` calls `tile_ir.lower_to_tile` and
`emit_ptx`, both of which raise `NotImplementedError` on unsupported tile
constructs (the cycle-14/15 loud-fail discipline documented in
`lower_ast.py` and elsewhere). The discipline requires the NIE to
propagate so the parent harness sees the real class and so a future
TIR-op subclass forces explicit dispatch instead of silent generic-error
aliasing.

**The sibling block 50 lines above already shows the correct pattern**
(`check.py:1662-1671`): re-raise `(NotImplementedError, AssertionError,
KeyboardInterrupt, SystemExit, MemoryError)` first, THEN catch
`Exception`. That site narrowed in restart 48/49. The two
`validate_kernel_tile_lowering` sites were missed by the same sweep and
remain bare-`Exception` at HEAD.

**Sibling sweep**:
| Site | Loud-fail re-raise? |
|---|---|
| `check.py:945-958, 974-983, 1017-1024` (`_proof_strict_effect_check`) | Yes (restart 49) |
| `check.py:1662-1671` (PTX full effect-check) | Yes (restart 49) |
| `check.py:1716-1723` (`-O1+` kernel tile validation) | NO — bare `except Exception` |
| `check.py:1744-1751` (`-O0` kernel tile validation) | NO — bare `except Exception` |
| `check.py:1834-1837, 1849-1859, 1872-1875` | NO at HEAD (uncommitted working-tree fix pending — see other Lane B findings) |
| `backend/ptx.py:1006-1011, 1052-1057` | Yes (restart 48 B2) |
| `frontend/autodiff_cli.py:58-68, 127-139` | Yes (restart 48 B3 / 49 B1) |
| `ir/lower_ast.py:643-664, 3082-3098` | Yes (restart 47 B1 / 49 B4) |

**Suggested fix**: Insert a re-raise clause before each
`except Exception as e:` at lines 1718 and 1746 matching the line-1662
pattern: `except (NotImplementedError, AssertionError, KeyboardInterrupt,
SystemExit, MemoryError): raise`.

**Suggested canary**:
`test_stage35_check_validate_kernel_tile_lowering_propagates_loud_fail`:
monkeypatch `helixc.backend.ptx.validate_kernel_tile_lowering` to raise
`NotImplementedError("synthetic-loud-fail")`; run `python -m helixc.check
src.hx` on a kernel-bearing source at `-O1` AND at `-O0` (parametrized
sub-cases). Assert the process returncode is non-zero AND
`NotImplementedError` appears in stderr (parent harness saw the
unhandled exception class). The current behavior produces `helixc: PTX
validation error: synthetic-loud-fail` with rc=1 and no class signal,
failing the assertion.

---

## B3 — `--emit-ptx` finalizer and `--emit-asm`/`-o` codegen blocks swallow loud-fail signals — MEDIUM

**File**: `helixc/check.py:1834-1837`, `:1849-1859`, `:1872-1875`
**Function**: `_main_inner` (the three artifact-emit branches that call
into the x86 / PTX backends)
**Bug family**: Parser / typechecker / codegen silent fallbacks — same
family as B2

**Issue at HEAD**: Three artifact-emit blocks each have a bare
`except Exception as e:` wrapping a backend call that can raise
`NotImplementedError`:

1. `--emit-asm` (line 1834): `compile_module_to_elf(mod)` →
   `_report_x86_codegen_exception(e)`. The reporter prints
   `helixc: internal error: <Type>: {e}` + `helixc: this is a compiler
   bug — please file an issue.` That tagline IS loud — but the NIE is
   still consumed in-process, never propagating to the harness. Mild.
2. `--emit-ptx` (line 1849): `kernel_only_module + lower_to_tile +
   emit_ptx`. Diagnostic is `   ptx: backend error: {e}` (with leading
   spaces, no helixc prefix, no class name, no compiler-bug tagline). The
   format is INCONSISTENT with both `--emit-asm` (compiler-bug tagline)
   and the line-1662 sibling (which prints `helixc: PTX validation
   error: <Type>: {e}` with the class name). Less loud than the x86
   sibling.
3. `-o` codegen write (line 1872): same as `--emit-asm` —
   `_report_x86_codegen_exception(e)`.

Sites 1 and 3 print the compiler-bug tagline (loud at the user level),
but ALL three sites swallow `NotImplementedError` /
`AssertionError` / `MemoryError` from the process boundary. Compare to
sibling at `ptx.py:1006-1011` which re-raises these classes before
generic-Exception fallback.

**Sibling sweep**:
| Site | Re-raise loud-fail? | Diagnostic format |
|---|---|---|
| `check.py:1662-1671` (PTX full eff-check) | Yes | `helixc: PTX validation error: <Type>: {e}` |
| `check.py:1834-1837` (`--emit-asm`) | No | `helixc: internal error: <Type>: {e}` + compiler-bug tagline |
| `check.py:1849-1859` (`--emit-ptx`) | No | `   ptx: backend error: {e}` (loose, no class, no tagline) |
| `check.py:1872-1875` (`-o` ELF write) | No | `helixc: internal error: <Type>: {e}` + compiler-bug tagline |
| `backend/ptx.py:1006-1011, 1052-1057` | Yes | `error: ptx: {e}` |
| `backend/x86_64.py:4383` (codegen) | n/a — `sys.exit(1)` after explicit print, no broad try-wrap | n/a |

**Suggested fix**: Insert `except (NotImplementedError, AssertionError,
KeyboardInterrupt, SystemExit, MemoryError): raise` before each
`except Exception as e:` at lines 1836, 1857, 1874. For line 1857
additionally consider routing through `_report_x86_codegen_exception` (or
a renamed `_report_codegen_exception`) so the user-facing tagline matches
the other two siblings.

**Suggested canary**: `test_stage35_check_emit_artifact_propagates_loud_fail`:
three parametrized cases — monkeypatch `compile_module_to_elf` to raise
NIE and invoke `helixc.check --emit-asm` and `helixc.check -o out.bin`;
monkeypatch `emit_ptx` to raise NIE and invoke `helixc.check
--emit-ptx`. In all three cases assert NIE class name appears in stderr
(or returncode signals the unhandled exception was seen).

**Working-tree status**: An uncommitted edit already adds the re-raise
clauses at all three sites. Fix sweep should verify + commit (not
re-author). The diagnostic-format inconsistency at line 1857 (loose
format, no class, no tagline) is NOT addressed by the pending edit —
leave the call open as part of B3's fix-sweep follow-up.

---

## B4 — `const_fold._try_fold_op` arithmetic blocks swallow loud-fail signals via the post-`FoldError` bare `except` — LOW

**File**: `helixc/ir/passes/const_fold.py:484`, `:516`, `:617`
**Function**: `_try_fold_op` (int ADD/SUB/MUL/DIV/MOD fold, float
ADD/SUB/MUL/DIV fold, SHL/SHR fold)
**Bug family**: Same as B2/B3 — bare `except Exception:` swallowing
loud-fail signals. Restart 50 B2 narrowed the SIBLING site
(`is_const`'s float-cast path, lines 367-370) for exactly this reason.

**Issue at HEAD**: Three try blocks each have the pattern:

```
try:
    ... Python arithmetic on int/float operands ...
except FoldError:
    raise              # restart 28.9 cycle 21 — trap 17001/17002 must surface
except Exception:
    return None        # un-foldable
```

The `except FoldError: raise` clause covers the explicit
FoldError/ShiftFoldError raises inside the try block. But the bare
`except Exception:` AFTER that re-raise still catches
`NotImplementedError` / `AssertionError` / `MemoryError` / `RecursionError`
silently and returns `None` (treats the op as un-foldable).

Today's blocks call only Python's built-in `+ - * / // % >> <<` on int
and float, which raise a fixed set (`OverflowError`, `ZeroDivisionError`,
`ValueError`, `TypeError`) — so no NIE is raised in practice. Risk is
purely latent: a future edit that calls into a helper raising NIE for an
unhandled TIR-op subclass would silently fall through to "un-foldable"
with no diagnostic, no progress report, just a fold-miss. This is the
same latent-bug shape that restart 50 B2 closed for `is_const`'s float
cast path.

**Sibling sweep**:
| Site | Narrow exception set |
|---|---|
| `const_fold.py:367-370` (`is_const` float cast) | `(ValueError, TypeError, OverflowError)` — restart 50 B2 |
| `const_fold.py:476-485` (int ADD/SUB/MUL/DIV/MOD) | `except Exception:` after `except FoldError: raise` — too broad |
| `const_fold.py:509-517` (float ADD/SUB/MUL/DIV) | same — too broad |
| `const_fold.py:609-618` (SHL/SHR) | same — too broad |

**Suggested fix**: Replace `except Exception:` with
`except (OverflowError, ZeroDivisionError, ValueError, TypeError):` at
all three sites, mirroring the restart 50 B2 narrowing pattern. The
`except FoldError: raise` clause is kept verbatim.

**Suggested canary**: `test_stage35_const_fold_propagates_loud_fail`:
fabricate a `CONST_INT` or `CONST_FLOAT` op whose `attrs["value"]` is
either an oddly-typed object that raises NIE on `+`, or monkeypatch
`_wrap_int_to_type` to raise `NotImplementedError("synthetic")`. Call
`_try_fold_op` directly; assert NIE propagates (currently silently
returns `None`).

---

## Cross-cutting clean areas confirmed at HEAD

- **Stale-artifact bad-invocation cleanup**:
  `_bad_invocation_cleanup_output` (x86_64.py:3971) called from all five
  `sys.exit(2)` paths in the flag parser (lines 4043, 4067, 4080, 4087).
  `_cleanup_bad_invocation_output` (check.py:1054) called from all six
  bad-invocation `return 2` paths. Pre-flag-parse exits (input is flag,
  output is flag, input == output) skip cleanup intentionally because no
  clean output path is determined. The pre-build
  `_remove_stale_output(sys.argv[2])` at x86_64.py:4142 runs once before
  the build pipeline begins, ensuring every subsequent `sys.exit(1)` from
  parse/typecheck/mono/lower leaves the requested output empty rather
  than stale. The `--help` short-circuit at `check.py:1040 /
  x86_64.py:4008 / ptx.py:815` does NOT clean output, intentionally
  (help is not a build).
- **Partial writes (atomic + os.replace + cleanup)**: All four production
  file-writers confirmed atomic:
    - `check.py:_atomic_write_bytes` (line 458) — mkstemp + os.replace +
      `except BaseException: cleanup; raise` (restart 46 B4).
    - `backend/x86_64.py:_atomic_write_output` (line 4090) — same.
    - `examples/run.py:96-112` — same (restart 46 B5).
    - `examples/dashboard_server.py:78-93` — same (restart 47 B5).
  No new production file-writers added since restart 47.
- **Backend / flag parity**: Verified the four CLIs accept the documented
  shared-flag set. The flag banner in `backend/x86_64.py:3999` and
  `backend/ptx.py:806` and the docstring in `check.py:19-44` all
  enumerate the same shared family (`--strict`, `--no-opt`, `-O0..-O3`,
  `--stdlib`, `--no-stdlib`, `-Wad=warn|error`, `-Wdeprecated=warn|error`,
  `-l <libname>`, `--no-color`, `--color`, `--hash`, `--hash-cons`).
  `check.py` additionally accepts emit-modes (`--emit-ast`, `--emit-ir`,
  `--emit-asm`, `--emit-ptx`, `--emit-proof-obligations`, `--doc`) and
  `--check-only`, which are check-specific by design and structurally do
  not apply to backends. No new check.py flags added since restart 48 —
  `--no-opt` parity is closed; the diff `5ee0362..7b945fa` shows no new
  flag introductions on either side.
- **CLI exit-code convention** (rc=0 help, rc=1 source/internal error,
  rc=2 bad invocation): Verified across all four CLIs.
  File-not-found is rc=2 uniformly (treated as bad-invocation because
  the path was supplied by the caller); LexError/ParseError/typecheck/
  mono/flatten errors are rc=1 uniformly; unknown flag is rc=2 uniformly;
  missing-input-path is rc=2 uniformly; `-h` / `--help` is rc=0 uniformly,
  prints to stdout, with a banner enumerating accepted flags (restart 49
  B2 + B3). One CLI gap found at HEAD: `autodiff_cli` did not reject
  unknown SHORT flags — see B1.
- **Bootstrap parser drift vs Python parser**: The brief references
  `helixc/bootstrap_parser.py` — that path does NOT exist at HEAD; the
  bootstrap parser was migrated to Helix-source
  `helixc/bootstrap/parser.hx` (7,647 lines) at Stage 30. The current
  Python parser is `helixc/frontend/parser.py`. Compared
  `bootstrap/parser.hx:skip_attributes` (lines 4430-4570) against
  `frontend/parser.py:_parse_attrs` (line 230 onward): bootstrap
  enumerates `@checkpoint`, `@deprecated`, `@since`, `@trace`,
  `@unwind`, `@kernel`, `@autotune` (Stage 33 alignment); Python parser
  is intentionally more permissive (accepts arbitrary `@<IDENT>` at line
  270). Drift direction is Python-broader, not bootstrap-stale. No new
  metadata kinds added in either parser since restart 47's verification.
- **`except Exception` narrowing across the compiler** (audit-wide
  sweep): Verified all sites; the open ones are B2, B3, B4 above.
  `lower_ast.py:643-664, 3082-3098` both correctly narrow;
  `backend/ptx.py:1006-1011, 1052-1057` both narrow (restart 48 B2);
  `frontend/autodiff_cli.py:58-68, 127-139` both narrow (restart 48 B3,
  49 B1); `frontend/autodiff.py:199-214` cache-bypass site is documented
  as deliberate graceful bypass with `_ad_warn` surfacing — acceptable;
  `frontend/diagnostics.py:74-77` defensive `isatty()` probe is allowed
  to swallow — acceptable; `frontend/autodiff.py:1296-1307` arithmetic
  simplifier is narrowed to `(OverflowError, ZeroDivisionError,
  ValueError, TypeError)`; `check.py:945-958, 974-983, 1017-1024,
  1662-1671` all narrow (restart 49).
- **Dev-only `if __name__ == "__main__":` runners**
  (`helixc/frontend/{lexer,parser,typecheck}.py`,
  `helixc/ir/lower_ast.py`, `helixc/frontend/presburger.py`): out of
  CLI-scope per the campaign brief — debug runners with no exit-code/
  help convention. Not audited for parity.

## Note on working-tree contamination

`git status` at audit time shows uncommitted modifications to
`helixc/check.py` and `helixc/frontend/autodiff_cli.py`. The diffs
implement fix candidates for B1 (autodiff_cli unknown short flag at
lines 91-101) and B3 (re-raise loud-fail at lines 1834-1837, 1849-1859,
1872-1875) — likely written by an earlier Lane B agent that ignored the
read-only constraint. Those fix candidates are correct in shape but
should be verified by the fix sweep before commit. B2
(`validate_kernel_tile_lowering` at 1716/1746) is NOT in the working
tree and remains an unaddressed finding at HEAD. B4 (`const_fold` bare
`except`) is likewise not in the working tree.
