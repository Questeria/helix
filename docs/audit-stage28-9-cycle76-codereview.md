# Audit Stage 28.9 cycle 76 — Code review

**Scope:** HEAD 92ffc5a (Stage 28.9 cycle-75 audit clean, 1/5 after cycle-73 reset).
**Mode:** STRICT READ-ONLY. No edits performed.
**Audit gate progress at entry:** 1/5 clean.

## Focus areas (rotation)

1. `helixc/check.py` — argument parser completeness, error-path coverage.
2. `helixc/stdlib/*.hx` — stale comments / TODO / FIXME without tracking ID dated 2026-04 or older.
3. `helixc/tests/` — brittle assertions or skip decorations that hide real failures (excluded: `test_codegen.py`, `test_autodiff.py`, `test_autodiff_reverse.py`).
4. TODO/FIXME/XXX/HACK comments older than 2026-04 with no tracking ID across the compiler.

## Findings

**0 findings at confidence >= 75%.**

## Sub-audit notes

### check.py (CLI driver)

- `parse_args` covers all documented flags: `--stdlib`, `--hash`, `--hash-cons`, `--strict`, `--check-only`, `--emit-ast`, `--emit-ir`, `--emit-asm`, `--emit-ptx`, `--doc`, `-O0..-O3`, `-o`, `-l<name>` (attached and separate), `-W<flag>[=<level>]`, `--no-color`, `--color`, `-h`/`--help`.
- Error paths: `-O` out-of-range emits `unknown opt level`; `-o`/`-l` missing arg emits explicit error; unknown long/short flag rejected; extra positional args reported.
- `main()` wrapper distinguishes user-environment errors (FileNotFound, Permission, IsADirectory, NotADirectory, UnicodeDecodeError → rc=2 with `helixc:` prefix) from internal errors (catch-all → "this is a compiler bug" with rc=1). AD-warning drain runs in `finally` and is itself wrapped so a drain failure cannot mask the primary failure.
- Pipeline gates surface diagnostics on stderr/stdout consistently (parse, typecheck, struct-mono, mod-flatten, impl-flatten, totality, deprecated, trace, panic, unwind, unsafe, autotune, fold/effect-check). `--check-only` short-circuit documented as intentional fast path (cycle-26 C25-1 NOTE).
- CLI tests in `tests/test_cli.py` cover all parser branches.

### stdlib/*.hx

- `grep -ni 'TODO|FIXME|XXX|HACK'` against `helixc/stdlib/` returned no matches.
- `grep -i 'deprecated|stale|obsolete|outdated|legacy|workaround|kludge|stub'` returned no matches.

### tests/

- Skip decorations: only three lines containing `pytest.skip(...)` exist, all in `test_codegen.py` (excluded scope) gating a single Stage-29 self-host loop closure work item; out of scope.
- `try/except: pass` patterns in `test_reflection.py:37-40` and `test_select_codegen.py:99-102` wrap `os.chmod(out_path, 0o755)` — platform-portability for Windows where chmod is a no-op. Not a hidden failure.
- No `pytest.importorskip`, `@pytest.mark.skipif`, `sys.platform` gating, `if False`, or silent `return # skip` patterns elsewhere.
- Hex-literal `assert` lines exist only in `test_codegen.py` (excluded).

### Repo-wide TODO/FIXME/XXX/HACK survey (compiler source)

`grep -n 'TODO|FIXME|XXX|HACK'` returned 8 hits. Per git blame:

| Location | Date | Tracking | Disposition |
|----------|------|----------|-------------|
| `backend/x86_64.py:2557` `TODO(stage30): ...` | 2026-05-10 | `(stage30)` tag + cycle 28.8 A7 context | OK |
| `frontend/ast_hash.py:392` `TODO follow-on cycle to recurse into GenericParam` | 2026-05-11 | embedded in cycle-20 invariant comment | OK |
| `ir/tile_ir.py:154` "TODO markers. The intent is to develop the real tiling rules" | 2026-05-03 | v0.1-Phase-0 milestone tag in docstring | OK |
| `ir/tile_ir.py:219` "(TODO real tiling rules)" | 2026-05-03 | same milestone | OK |
| `backend/ptx.py:332` `// TODO: {op.kind.value}` | 2026-05-03 | generated PTX marker emitted for unhandled ops; not a source TODO | OK |
| `tests/test_codegen.py:1867` | excluded scope | n/a | excluded |
| `tests/test_ptx.py:140-141` | assertion verifying trap-strings ABSENT from output | n/a | not a TODO |

None pre-date 2026-04. All have either an explicit tracking tag (stage30) or sit inside a cycle-numbered audit-narrative block; none qualify as "older than 2026-04 with no tracking ID."

## Verdict

**PASS.**

0 findings at confidence >= 75%. Stage 28.9 audit gate advances 2/5.

## Procedural attestation

No `Edit` calls made. Only `Read`, `Glob`, `Grep`, `Bash` (read-only: `git log`, `git blame`), and exactly one `Write` (this file). C1-C75 + deferred-known findings not re-flagged.
