"""
helixc/frontend/autodiff_cli.py — print the symbolic derivative of a function.

Usage:
    python -m helixc.frontend.autodiff_cli <file.hx> <function_name> [<var_name>]
    python -m helixc.frontend.autodiff_cli --dump-ast-hashes <file.hx>

If <var_name> is omitted, differentiates w.r.t. the first parameter.

--dump-ast-hashes prints `<fn_name> : <12-char hex hash>` for every top-level
fn in the file. The hash is the structural (alpha-equivalent) hash of the
fn AST and is stable across runs.

Example:
    $ cat loss.hx
    fn loss(x: f32) -> f32 { x * x }

    $ python -m helixc.frontend.autodiff_cli loss.hx loss
    d(loss)/d(x) = (x + x)

License: Apache 2.0
"""

from __future__ import annotations

import sys

from .parser import parse
from . import ast_nodes as A
from .autodiff import differentiate, fmt
from .ast_hash import structural_hash, short_hash


# Restart 47 B3: structured error handling matching helixc.check /
# helixc.backend.x86_64 / helixc.backend.ptx — surface OSError, parse errors,
# and internal AD errors as one-line diagnostics rather than raw Python
# tracebacks.
def _read_source(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"error: autodiff_cli: cannot read {path}: not found",
              file=sys.stderr)
        sys.exit(2)
    except OSError as e:
        print(f"error: autodiff_cli: cannot read {path}: {e}",
              file=sys.stderr)
        sys.exit(2)


def _parse_or_exit(src: str, path: str):
    # Restart 48 B3: preserve loud-fail discipline. NotImplementedError
    # from parser.parse() signals a TyNode/ASTNode subclass that needs
    # explicit dispatch — must propagate, not be flattened to "parse
    # error: ...". Mirrors restart 47 B1's narrowing of
    # lower_ast._resolve_monomorphized_struct_type.
    try:
        return parse(src)
    except (NotImplementedError, AssertionError, KeyboardInterrupt,
            SystemExit, MemoryError):
        raise
    except Exception as e:
        # Restart 49 B1: parse error is a SOURCE error → rc=1 (matches
        # check.py / x86_64.py / ptx.py convention). Bad invocation is
        # rc=2; runtime/internal errors are rc=1.
        print(f"error: autodiff_cli: parse: {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _dump_ast_hashes(path: str) -> int:
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for it in prog.items:
        if isinstance(it, A.FnDecl):
            print(f"{it.name} : {short_hash(structural_hash(it))}")
    return 0


def main():
    # Restart 49 B2: accept -h / --help → print docstring to stdout, exit 0
    # (matches check.py UX convention).
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(__doc__.strip())
        sys.exit(0)
    # Restart 49 B1: bad invocation (no args) is rc=2 to match the convention
    # established by check.py / x86_64.py / ptx.py.
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    # Restart 51 B1: reject unknown single-dash flags (e.g. -O1, -Wad=error)
    # before the partition below silently consumes them as positional args.
    # Matches the unknown-flag-rc=2 convention of check.py / x86_64.py / ptx.py.
    _single_dash_unknowns = [
        a for a in sys.argv[1:]
        if a.startswith("-") and not a.startswith("--") and a != "-h"
    ]
    if _single_dash_unknowns:
        for uf in _single_dash_unknowns:
            print(f"error: autodiff_cli: unknown flag {uf}", file=sys.stderr)
        sys.exit(2)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if "--dump-ast-hashes" in flags:
        if len(args) < 1:
            print("usage: --dump-ast-hashes <file.hx>", file=sys.stderr)
            # Restart 49 B1: bad-invocation rc=2.
            sys.exit(2)
        sys.exit(_dump_ast_hashes(args[0]))

    if len(sys.argv) < 3 or len(args) < 2:
        print(__doc__.strip(), file=sys.stderr)
        # Restart 49 B1: bad-invocation rc=2.
        sys.exit(2)

    path = args[0]
    fn_name = args[1]
    var = args[2] if len(args) > 2 else None
    emit_function = "--as-function" in flags

    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    target = None
    for it in prog.items:
        if isinstance(it, A.FnDecl) and it.name == fn_name:
            target = it
            break
    if target is None:
        print(f"error: function {fn_name!r} not found in {path}",
              file=sys.stderr)
        sys.exit(1)
    if not target.params:
        print(f"error: function {fn_name!r} has no parameters", file=sys.stderr)
        sys.exit(1)

    differentiate_var = var or target.params[0].name
    try:
        deriv = differentiate(target.body, differentiate_var)
    # Restart 48 B3: preserve loud-fail discipline (same precedent as
    # _parse_or_exit above).
    except (NotImplementedError, AssertionError, KeyboardInterrupt,
            SystemExit, MemoryError):
        raise
    except Exception as e:
        # Restart 49 B1: differentiate runtime failures are SOURCE/INTERNAL
        # errors → rc=1 (matches check.py / x86_64.py / ptx.py convention
        # for internal-error exits). Bad-invocation only is rc=2.
        print(f"error: autodiff_cli: differentiate: {e}", file=sys.stderr)
        sys.exit(1)

    if emit_function:
        # Restart 50 B1: preserve source param types and return type instead
        # of hardcoding f32. The previous hardcode produced type-wrong
        # `fn loss__grad(x: f32) -> f32` for an `fn loss(x: f64) -> f64`
        # source, breaking the QUICKSTART round-trip example for any
        # non-f32 source. Format the type from the AST node if it has a
        # `name` attribute; fall back to f32 only for D<T>-wrapped or
        # otherwise non-printable types.
        def _format_ty(ty) -> str:
            # TyName, TyScalar, TyPrim, etc. typically expose `.name`.
            name = getattr(ty, "name", None)
            if isinstance(name, str) and name:
                return name
            # D<T> wrappers (TyGeneric base="D", args=[T]) unwrap to inner.
            base = getattr(ty, "base", None)
            args = getattr(ty, "args", None)
            base_name = getattr(base, "name", None)
            if base_name == "D" and args and len(args) == 1:
                return _format_ty(args[0])
            # Fallback: keep f32 (legacy behavior) so output stays
            # parseable rather than emitting `?` or a repr() string.
            return "f32"
        params_str = ", ".join(
            f"{p.name}: {_format_ty(p.ty)}" for p in target.params
        )
        ret_str = _format_ty(target.return_ty) if getattr(target, "return_ty", None) is not None else "f32"
        print(f"fn {fn_name}__grad({params_str}) -> {ret_str} {{")
        print(f"    {fmt(deriv)}")
        print(f"}}")
    else:
        print(f"d({fn_name})/d({differentiate_var}) = {fmt(deriv)}")


if __name__ == "__main__":
    main()
