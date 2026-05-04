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


def _dump_ast_hashes(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    prog = parse(src)
    for it in prog.items:
        if isinstance(it, A.FnDecl):
            print(f"{it.name} : {short_hash(structural_hash(it))}")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if "--dump-ast-hashes" in flags:
        if len(args) < 1:
            print("usage: --dump-ast-hashes <file.hx>", file=sys.stderr)
            sys.exit(1)
        sys.exit(_dump_ast_hashes(args[0]))

    if len(sys.argv) < 3 or len(args) < 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    path = args[0]
    fn_name = args[1]
    var = args[2] if len(args) > 2 else None
    emit_function = "--as-function" in flags

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    prog = parse(src)
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
    deriv = differentiate(target.body, differentiate_var)

    if emit_function:
        # Emit a complete Helix function definition: fn <name>__grad(...) -> ... {
        #     <expr>
        # }
        # Use the same parameter list as the source (minus D wrappers, since
        # the gradient takes plain floats).
        params_str = ", ".join(
            f"{p.name}: f32" for p in target.params
        )
        ret_str = "f32"
        print(f"fn {fn_name}__grad({params_str}) -> {ret_str} {{")
        print(f"    {fmt(deriv)}")
        print(f"}}")
    else:
        print(f"d({fn_name})/d({differentiate_var}) = {fmt(deriv)}")


if __name__ == "__main__":
    main()
