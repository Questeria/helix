# Audit Stage 35 Restart 50 — Lane B (compiler / backend / CLI)

## Summary

Lane B (compiler / backend / CLI) inspected against the restart-46/47/48/49
hardening discipline. **5 findings** — all are sibling-pattern violations of
the loud-fail discipline established by restart 48 B2 (re-raise
`NotImplementedError` / `AssertionError` instead of swallowing under
`except Exception`).

- **HIGH ×4** — `validate_kernel_tile_lowering` + final-codegen sites in
  `check.py` + `x86_64.py` silently downgrade `NotImplementedError` to
  `rc=1` with `{e}`-only diagnostics. Type-name + "compiler bug" tagline
  are both lost. A future TIR/AST kind would silently miscompile under
  any of these paths.
- **MEDIUM ×3** — `check.py` strict-effect-check wrapper trio surfaces the
  type-name but mislabels NotImplementedError as "strict-effect-check:
  ERROR", confusing users about whether the failure is internal vs source.

**Verdict: DIRTY** (1 HIGH bug-family with 4 instances + 1 MEDIUM
bug-family with 3 instances).

---

## Family 1: Loud-fail discipline violations in PTX-validation +
codegen paths (HIGH)

**Pattern.** Restart 48 B2 narrowed `helixc/backend/ptx.py`'s outer
`except Exception` to re-raise `(NotImplementedError, AssertionError,
KeyboardInterrupt, SystemExit, MemoryError)` so a future AST/TIR
subclass forces explicit dispatch instead of silently falling through to
`error: ptx: <msg>` (rc=1). Restart 48 B3 did the same for
`autodiff_cli.py`. **The sibling sweep missed four equivalent sites in
`check.py` and `x86_64.py`.**

### Site table

| # | File:line | Call | Diagnostic on swallow | Severity |
|---|-----------|------|----------------------|----------|
| F1a | `helixc/check.py:1845` | `emit_ptx(tile_mod)` (`--emit-ptx` branch) | `   ptx: backend error: {e}` | HIGH |
| F1b | `helixc/check.py:1706` | `validate_kernel_tile_lowering(mod)` (opt-pipeline) | `helixc: PTX validation error: {e}` | HIGH |
| F1c | `helixc/check.py:1734` | `validate_kernel_tile_lowering(mod)` (-O0 branch) | `helixc: PTX validation error: {e}` | HIGH |
| F1d | `helixc/backend/x86_64.py:4325` | `validate_kernel_tile_lowering(mod)` (opt path) | `error: ptx: {e}` | HIGH |
| F1e | `helixc/backend/x86_64.py:4342` | `validate_kernel_tile_lowering(mod)` (no-opt path) | `error: ptx: {e}` | HIGH |
| F1f | `helixc/backend/x86_64.py:4383` | `compile_module_to_elf(mod)` (final codegen) | `error: codegen: {e}` | HIGH |

(F1f counts as one HIGH because the final-codegen path is the largest
loud-fail surface — any new TIR op visited by the x86 emitter funnels
through it.)

### Empirical confirmation for F1a

```
$ python -c "
from unittest.mock import patch
def boom(*a, **kw): raise NotImplementedError('synthetic new TIR node kind')
from helixc.backend import ptx as ptx_mod
from helixc.check import main
with patch.object(ptx_mod, 'emit_ptx', boom):
    rc = main(['<kernel.hx>', '--emit-ptx', '--no-stdlib'])
print('rc=', rc)
"
[...]
   typecheck: OK
   totality:  OK
helixc: PTX validation error: synthetic new TIR node kind   <-- type-name lost
rc= 1                                                       <-- silent downgrade
```

Compare with the clean restart-48 path for `python -m helixc.backend.ptx`:
the same monkey-patch would propagate the NotImplementedError to Python
with a traceback (loud-fail), forcing the maintainer to add explicit
dispatch.

### Clean sibling sites already correct

For reference (do NOT touch):

| File:line | Why clean |
|-----------|-----------|
| `helixc/backend/ptx.py:1006-1011` | Restart 48 B2: re-raises loud-fail set before `except Exception`. |
| `helixc/backend/ptx.py:1052-1057` | Restart 48 B2: outer try same pattern. |
| `helixc/frontend/autodiff_cli.py:60-68` | Restart 48 B3: `_parse_or_exit`. |
| `helixc/frontend/autodiff_cli.py:131-139` | Restart 48 B3: `differentiate(...)` wrapper. |
| `helixc/check.py:1824-1825` | Routes through `_report_x86_codegen_exception` which preserves type-name + "compiler bug" tagline. |
| `helixc/check.py:1862-1863` | Same — `_report_x86_codegen_exception`. |
| `helixc/ir/lower_ast.py:660` | Restart 47 B1: narrowed to `(KeyError, AttributeError)`. |
| `helixc/ir/lower_ast.py:3097` | Restart 49 B4: narrowed to `(KeyError, AttributeError, TypeError, ValueError)`. |

### Regression test needed (per family)

```python
def test_stage35_check_emit_ptx_propagates_not_implemented(monkeypatch):
    """check.py --emit-ptx must NOT swallow NotImplementedError from
    emit_ptx / lower_to_tile / kernel_only_module — that defeats the
    restart-48 B2 loud-fail discipline at a sibling site."""
    import helixc.backend.ptx as ptx_mod
    def boom(*a, **kw): raise NotImplementedError("new TIR kind")
    monkeypatch.setattr(ptx_mod, "emit_ptx", boom)
    from helixc.check import main
    # Test passes when this raises NotImplementedError OR when stderr
    # contains "internal error" + "NotImplementedError" + "compiler bug".
    ...

def test_stage35_x86_codegen_propagates_not_implemented(monkeypatch):
    """`helixc.backend.x86_64` final compile_module_to_elf wrapper must
    re-raise NotImplementedError / AssertionError instead of flattening
    to `error: codegen: <msg>` with rc=1."""
    ...

def test_stage35_validate_kernel_tile_lowering_propagates_loud_fail():
    """All four call sites of validate_kernel_tile_lowering (check.py
    1706/1734 + x86_64.py 4325/4342) must re-raise NotImplementedError."""
    ...
```

Source-text invariant test (cheap):

```python
def test_stage35_check_py_loud_fail_around_ptx_emit_and_validate():
    """check.py + x86_64.py must re-raise the loud-fail set before any
    except Exception that wraps emit_ptx / validate_kernel_tile_lowering /
    compile_module_to_elf."""
    import inspect, helixc.check, helixc.backend.x86_64
    for mod in (helixc.check, helixc.backend.x86_64):
        src = inspect.getsource(mod)
        needle = "except (NotImplementedError, AssertionError, KeyboardInterrupt,"
        assert src.count(needle) >= 3, (
            f"{mod.__name__} should have at least 3 narrowed handlers "
            f"around PTX-emit / validate / codegen wrappers; "
            f"found {src.count(needle)} occurrences"
        )
```

### Recommended fix shape (per site)

Insert before each offending `except Exception`:

```python
except (NotImplementedError, AssertionError, KeyboardInterrupt,
        SystemExit, MemoryError):
    raise
```

For F1f (x86_64.py `compile_module_to_elf` wrapper), also consider
mirroring `_report_x86_codegen_exception`'s discrimination of
`ValueError("module has no function ")` (user error, no tagline) vs
other exceptions (compiler bug, with tagline). Today the standalone
backend CLI prints `error: codegen: module has no function 'main'`
which is correct UX but groups it with internal errors.

---

## Family 2: strict-effect-check broad excepts attenuate loud-fail signals (MEDIUM)

**Pattern.** `helixc/check.py` has three sibling `except Exception` blocks
inside `_compute_strict_effects` (the `--strict` wrapper) that catch the
loud-fail set but DO surface `type(e).__name__` in the diagnostic. So
NotImplementedError is partially preserved (the user sees the type name)
but mis-labeled as a "strict-effect-check: ERROR" — implying it's a user
problem in their declared effects, when it's actually an unimplemented
AST/TIR kind.

### Site table

| # | File:line | Call | Diagnostic on swallow | Severity |
|---|-----------|------|----------------------|----------|
| F2a | `helixc/check.py:949` | `grad_pass(strict_prog)` + `lower(strict_prog)` | `strict-effect-check: ERROR\n     {type}: {msg}` | MEDIUM |
| F2b | `helixc/check.py:971` | `fold_module/cse_module/dce_module/fdce_module` | same | MEDIUM |
| F2c | `helixc/check.py:1011` | `effect_check_module(mod)` post-opt | same | MEDIUM |
| F2d | `helixc/check.py:1653` | `lower + effect_check_module` (--emit-ptx full path) | `helixc: PTX validation error: {type}: {msg}` | MEDIUM |

All four are tagged "Audit 28.8" — they predate restart 47/48's loud-fail
push. The diagnostic format leaks the type-name, so a future
NotImplementedError is *visible* but mis-categorized as a strict-effect
or PTX-validation error rather than an internal compiler bug.

### Recommended fix shape

Same pattern as Family 1 — insert the loud-fail re-raise before each
`except Exception`:

```python
except (NotImplementedError, AssertionError, KeyboardInterrupt,
        SystemExit, MemoryError):
    raise
except Exception as e:
    msg = (
        f"strict-effect-check: ERROR\n"
        f"     {type(e).__name__}: {e}"
    )
    ...
```

The outer `main()` wrapper at `check.py:561` will then format these as
`helixc: internal error: NotImplementedError: ...` + `helixc: this is
a compiler bug — please file an issue.` — the correct "compiler bug"
UX. Severity is MEDIUM rather than HIGH because the type-name does
appear in the current diagnostic, so a grep-savvy user could still
identify the loud-fail; but the rc convention (1 = source error)
mis-classifies what is actually an internal-error condition (which
the outer wrapper handles with the same rc=1 anyway, so the
rc-bucket is unchanged — only the *message* changes).

### Regression test needed

```python
def test_stage35_strict_effect_check_propagates_not_implemented(monkeypatch):
    import helixc.frontend.grad_pass as gp
    def boom(prog): raise NotImplementedError("new AST kind")
    monkeypatch.setattr(gp, "grad_pass", boom)
    ...assert "internal error" in stderr
    ...assert "NotImplementedError" in stderr
    ...assert "compiler bug" in stderr     # <-- key delta from current
```

---

## Areas verified clean

### Stale `-o` artifact cleanup

All `-o`-touching return paths in `helixc/check.py` are protected by
`_remove_stale_output(a.output)` BEFORE any source-level work begins
(lines 1124 and 1148), and the final write at line 1865 goes through
`_atomic_write_bytes`. No intermediate return path between
1148 and 1865 produces an output artifact, so no stale leakage between
those points.

`helixc/backend/x86_64.py` calls `_bad_invocation_cleanup_output()` on
every bad-invocation path (lines 4043, 4067, 4080, 4087). Input-flag
and source==output check paths skip cleanup correctly (the function is
a no-op when input is a flag, and would be destructive when source ==
output).

### Atomic-write parity

Confirmed clean per restart 47 scan:

| Site | Pattern |
|------|---------|
| `check.py:_atomic_write_bytes` (445-483) | mkstemp + chmod + os.replace + BaseException cleanup |
| `x86_64.py:_atomic_write_output` (4090-4117) | same |
| `examples/dashboard_server.py` (74-93) | same |
| `examples/run.py` (92-112) | same |

### CLI flag parity (check / x86_64 / ptx / autodiff_cli)

All four CLIs accept the restart-46 (-O0..-O3, --no-opt) +
restart-47 (-l <name> / -l<name>, --no-color/--color, --hash/--hash-cons)
+ restart-49 (-h/--help, rc=2 bad-invocation, rc=1 source error)
contract. Verified per `test_stage35_cli_help_flag_works_and_exits_zero`
parametrized matrix (8/8 pass).

Backend-only modes (`--emit-ast/--emit-ir/--emit-asm/--emit-ptx/--doc/
--check-only/--emit-proof-obligations`) live only in `check.py`; backends
don't expose them — correct mutual-exclusion.

### Exit-code convention

- `check.py`: rc=2 (bad invocation) at 1081, 1092, 1103, 1115, 1133,
  1136, 1138, 1145, 1161, 1163, 1171, 1188, 1244; rc=1 (compile error)
  on all pipeline-phase failures. Consistent.
- `x86_64.py`: rc=2 on bad-invocation (4011, 4017, 4021, 4028, 4044,
  4068, 4081, 4088); rc=1 on compile failures (4334, 4376, 4385, 4389).
  Consistent.
- `ptx.py`: rc=2 on bad-invocation (820, 866, 879, 885, 891, 894,
  901, 904, 922); rc=1 on compile failures (910, 916, 930, 935, 938,
  941, 946, 963, 975, 989, 1011, 1042, 1057). Consistent.
- `autodiff_cli.py`: restart-49 fix verified clean (45, 49, 90, 98,
  104, 121, 124).

### Parser AST node kinds

`git log 8c80731..HEAD -- helixc/frontend/ast_nodes.py
helixc/frontend/parser.py helixc/frontend/ast_hash.py
helixc/bootstrap/parser.hx` returns **empty** since the Stage 33 close
commit (`8c80731 Close Stage 33`). No drift; bootstrap parser still
aligned per restart 47 verification.

### `helixc/ir/lower_ast.py`

All `except` handlers narrow correctly:
- line 373: `except ValueError` around `list.index()` (correct).
- line 660: `except (KeyError, AttributeError)` (restart 47 B1).
- line 3032: `except ValueError` around `list.index()` (correct).
- line 3097: `except (KeyError, AttributeError, TypeError, ValueError)`
  (restart 49 B4).

No broad `except Exception` remains in `lower_ast.py`. The
"swallow a structural-hash divergence" concern from the hunt list is
already addressed at the sole structural_hash call site (3079-3097).

### `helixc/ir/passes/const_fold.py` broad excepts

Three `except Exception: return None` sites at lines 363, 478, 510, 611,
each preceded by `except FoldError: raise` to preserve the trap-17001/
17002 contract. Per restart-47 Audit 28.9 cycle 21 fix (already in
place). These wrap arithmetic primitives + `_source_widen_int` +
`_is_unsigned_int_type` + `_binary_int_bits` + `_unsigned_domain` —
all narrow utility functions where NotImplementedError is structurally
unreachable. Tolerable as-is; if a future edit moves a richer call
inside the try, the loud-fail re-raise pattern should be added (cost
~3 lines per site).

### Type-design issues (tagged unions / discriminated types)

No catch-all branches in `CliArgs`, `tir.OpKind`, `A.TyNode` or related
discriminated types observed silently widening. `_resolve_monomorphized_struct_type`
(lower_ast.py:643) and the structural_hash dispatch are both fail-loud
on unknown subclasses per restart 47/49 fixes. The `_KNOWN_LONG_FLAGS` /
`_KNOWN_WARNING_NAMES` frozensets in `check.py` properly reject unknown
flags (line 307: `errors.append(f"unknown flag: {tok}")`).

### Test coverage

Restart 49 B2-B4 + restart 48 B2-B3 tests (`test_stage35_*` family in
`helixc/tests/test_cli.py`) verified passing — 120 stage35-tagged tests
pass on HEAD (64.66s).

---

## Restart-50 verdict

**DIRTY** — 4 HIGH + 3 MEDIUM sibling violations of the restart-48 B2/B3
loud-fail discipline. The standalone `helixc.backend.ptx` and
`helixc.frontend.autodiff_cli` CLIs are correctly hardened; the wrapper
sites in `helixc.check` (`--emit-ptx` branch + PTX-validation calls +
strict-effect-check wrappers) and `helixc.backend.x86_64` (PTX-validation
calls + final codegen wrapper) are not. Same bug family; same one-line
fix per site (insert the `(NotImplementedError, AssertionError, ...):
raise` branch before each `except Exception`).

This counts as one bug-family (loud-fail discipline) with seven
instances, plus zero other findings. Gate counter should NOT advance.
