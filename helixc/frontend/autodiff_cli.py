"""
helixc/frontend/autodiff_cli.py — print the symbolic derivative of a function.

Usage:
    python -m helixc.frontend.autodiff_cli <file.hx> <function_name> [<var_name>]

If <var_name> is omitted, differentiates w.r.t. the first parameter.

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


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if len(args) < 2:
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
