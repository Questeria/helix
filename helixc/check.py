"""
helixc/check.py — `python -m helixc.check <file.hx>` developer CLI.

Runs the front-end pipeline on a Helix source file:
1. Lex + parse
2. Typecheck (with did-you-mean suggestions)
3. Totality (structural recursion check; @partial fns skipped)
4. Optional: --hash flag prints structural hash for each top-level fn
5. Optional: --stdlib flag bundles helixc/stdlib/transcendentals.hx into the parse

Exits 0 if clean, nonzero with diagnostics on any failure.

Examples:
    python -m helixc.check loss.hx
    python -m helixc.check --hash loss.hx
    python -m helixc.check --stdlib loss.hx

License: Apache 2.0
"""

from __future__ import annotations

import sys
import os

from .frontend.parser import parse, ParseError
from .frontend.typecheck import typecheck
from .frontend.totality import check_totality
from .frontend.ast_hash import structural_hash, short_hash
from .frontend import ast_nodes as A


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    flags: set[str] = set()
    while args and args[0].startswith("--"):
        flags.add(args.pop(0))
    if not args:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path = args[0]
    if not os.path.exists(path):
        print(f"helixc-check: file not found: {path}", file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    print(f"-- helixc-check: {path}")

    # 1. Parse
    try:
        prog = parse(src, include_stdlib=("--stdlib" in flags))
    except ParseError as e:
        # Use render() with source for caret display, fall back to bare
        # str(e) if the parse error doesn't have render().
        rendered = e.render(source=src, filename=path) \
            if hasattr(e, "render") else str(e)
        print(f"PARSE ERROR:", file=sys.stderr)
        for line in rendered.splitlines():
            print(f"  {line}", file=sys.stderr)
        return 1
    fn_count = sum(1 for it in prog.items if isinstance(it, A.FnDecl))
    print(f"   parse:    OK  ({fn_count} fns, {len(prog.items)} items)")

    # 2. Typecheck
    errs = typecheck(prog)
    if errs:
        print(f"   typecheck: {len(errs)} ERRORS")
        for e in errs[:20]:
            # Render with source-line + caret if the error has a render()
            # method (it does — see TypeError_.render).
            rendered = e.render(source=src, filename=path) \
                if hasattr(e, "render") else str(e)
            for line in rendered.splitlines():
                print(f"     {line}")
        if len(errs) > 20:
            print(f"     ... and {len(errs) - 20} more")
        return 1
    print(f"   typecheck: OK")

    # 3. Totality
    fails = check_totality(prog)
    if fails:
        print(f"   totality:  {len(fails)} fn(s) NOT proven total")
        for name, reason in fails:
            print(f"     {name}: {reason}")
        if "--strict" in flags:
            return 1
    else:
        print(f"   totality:  OK")

    # 4. Optional hash dump
    if "--hash" in flags:
        print(f"   hashes:")
        for it in prog.items:
            if isinstance(it, A.FnDecl):
                print(f"     {it.name:<40} {short_hash(structural_hash(it))}")

    # 5. Optional IR dump for parity / debugging.
    if "--emit-ir" in flags:
        from .ir.lower_ast import lower
        mod = lower(prog)
        print(f"   ir:")
        for fn in mod.functions.values():
            print(f"     fn {fn.name}:")
            for blk in fn.blocks:
                print(f"       block {blk.id}:")
                for op in blk.ops:
                    operands = ",".join(str(o.id) for o in op.operands)
                    results = ",".join(str(r.id) for r in op.results)
                    attrs_str = (" " + str(dict(op.attrs))) if op.attrs else ""
                    print(f"         {op.kind.name} ({operands}) -> ({results}){attrs_str}")

    print(f"-- clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
