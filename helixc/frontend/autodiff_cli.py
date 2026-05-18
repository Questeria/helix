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


def _print_program_hash(path: str) -> int:
    """Stage 58 / Tier 4 #13 CLI exposure: print the content-addressed
    program_hash of the parsed program. Span-independent + alpha-
    equivalence-aware. Use case: detect whether a source file's
    semantic content has changed between commits (vs. whitespace/
    comment-only diffs)."""
    from .ast_hash import program_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    print(program_hash(prog))
    return 0


def _check_program_hash(path: str, expected: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: assertion-style hash
    check for CI / git pre-commit hooks.

    Computes program_hash(parse(path)) and compares to the
    `expected` argument. Accepts either the full 64-hex or the
    12-hex short form (compares prefix-match).

    Exit codes:
      0 — match (silent)
      1 — mismatch (prints expected vs actual)
      2 — bad arg (parse failure, unreachable file, etc.)

    Use case: in a CI script, ensure a critical-file's semantic
    content hasn't drifted unexpectedly. Pre-commit hook example:
      python -m helixc.frontend.autodiff_cli --check-program-hash \\
          stdlib/result.hx 0d03c61f9975 || {
              echo "stdlib/result.hx drifted"; exit 1;
          }
    """
    from .ast_hash import program_hash, short_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    actual = program_hash(prog)
    if actual == expected:
        return 0
    if len(expected) == 12 and short_hash(actual) == expected:
        return 0
    print(f"hash mismatch")
    print(f"  expected: {expected}")
    print(f"  actual:   {short_hash(actual)} ({actual})")
    return 1


def _list_fns(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate all FnDecls
    in a source file with their signature + body hashes side by side.

    Output format per fn (one line):
      `<name> sig=<12hex> body=<12hex>`
    Sorted alphabetically by fn name for stable diff-friendly output.

    Use case:
    - Repository exploration / inventory
    - Pre-compute hash table for a build cache without subprocess
      cost per fn
    - Quick visual scan for fns whose body changed (sig matches but
      body differs)

    Exit 0 always (no failure mode beyond parse error).
    """
    from .ast_hash import structural_hash, fn_signature_hash, short_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    fns = sorted(
        (it for it in prog.items if isinstance(it, A.FnDecl)),
        key=lambda f: f.name,
    )
    for fn in fns:
        sig = short_hash(fn_signature_hash(fn))
        body = short_hash(structural_hash(fn))
        print(f"{fn.name} sig={sig} body={body}")
    return 0


def _fn_sig_hash(path: str, fn_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print the signature-only
    hash of a specific fn. Use case: detect whether a fn's PUBLIC
    contract changed between commits (callers need recompile) vs
    body-only refactor (internal-only, no caller impact).

    Returns exit 0 on success, 1 if the fn name is not found in
    the source file.
    """
    from .ast_hash import fn_signature_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for it in prog.items:
        if isinstance(it, A.FnDecl) and it.name == fn_name:
            print(fn_signature_hash(it))
            return 0
    print(f"error: autodiff_cli: fn {fn_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _changed_fns(path_a: str, path_b: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: list FnDecls that
    changed between two source files at the AST-hash level.

    Prints per-changed-fn lines:
      `+name : <hex12>`   — added in b (not in a)
      `-name : <hex12>`   — removed in a (not in b)
      `~name : <a12> -> <b12>` — present in both but body/sig changed

    Exit code:
      0 — no changes (or only unchanged fns)
      1 — at least one changed fn

    Use case: PR review — quickly identify which functions changed
    semantically between baseline + branch, ignoring whitespace /
    bound-variable renames / comment edits. Complements
    --diff-program-hash (whole-program YES/NO) with fn-granular
    breakdown.
    """
    from .ast_hash import structural_hash, short_hash
    src_a = _read_source(path_a)
    src_b = _read_source(path_b)
    prog_a = _parse_or_exit(src_a, path_a)
    prog_b = _parse_or_exit(src_b, path_b)
    a_fns = {it.name: structural_hash(it) for it in prog_a.items
             if isinstance(it, A.FnDecl)}
    b_fns = {it.name: structural_hash(it) for it in prog_b.items
             if isinstance(it, A.FnDecl)}
    a_names = set(a_fns.keys())
    b_names = set(b_fns.keys())
    changed = 0
    # Added fns (in b but not in a).
    for name in sorted(b_names - a_names):
        print(f"+{name} : {short_hash(b_fns[name])}")
        changed += 1
    # Removed fns (in a but not in b).
    for name in sorted(a_names - b_names):
        print(f"-{name} : {short_hash(a_fns[name])}")
        changed += 1
    # Modified fns (in both with different hash).
    for name in sorted(a_names & b_names):
        if a_fns[name] != b_fns[name]:
            print(f"~{name} : {short_hash(a_fns[name])} -> "
                  f"{short_hash(b_fns[name])}")
            changed += 1
    return 0 if changed == 0 else 1


def _diff_program_hash(path_a: str, path_b: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: compare program_hash of
    two source files. Prints `MATCH` + exits 0 when semantically
    equivalent (same AST hash modulo span/alpha-equivalence), or
    `DIFFER\\n  a=...\\n  b=...` + exits 1 when they diverge.

    Use case: CI sanity check that a refactor PR is semantically
    identical to the baseline (e.g., formatter-only diff, var
    rename, etc.). Build cache: if hashes match, skip recompile.
    """
    from .ast_hash import program_hash, short_hash
    src_a = _read_source(path_a)
    src_b = _read_source(path_b)
    prog_a = _parse_or_exit(src_a, path_a)
    prog_b = _parse_or_exit(src_b, path_b)
    ha = program_hash(prog_a)
    hb = program_hash(prog_b)
    if ha == hb:
        print(f"MATCH {short_hash(ha)}")
        return 0
    print("DIFFER")
    print(f"  a={short_hash(ha)} ({path_a})")
    print(f"  b={short_hash(hb)} ({path_b})")
    return 1


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

    if "--program-hash" in flags:
        if len(args) < 1:
            print("usage: --program-hash <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_print_program_hash(args[0]))

    if "--diff-program-hash" in flags:
        if len(args) < 2:
            print("usage: --diff-program-hash <a.hx> <b.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_diff_program_hash(args[0], args[1]))

    if "--changed-fns" in flags:
        if len(args) < 2:
            print("usage: --changed-fns <a.hx> <b.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_changed_fns(args[0], args[1]))

    if "--fn-sig-hash" in flags:
        if len(args) < 2:
            print("usage: --fn-sig-hash <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_sig_hash(args[0], args[1]))

    if "--list-fns" in flags:
        if len(args) < 1:
            print("usage: --list-fns <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_fns(args[0]))

    if "--check-program-hash" in flags:
        if len(args) < 2:
            print("usage: --check-program-hash <file.hx> "
                  "<expected_hex_hash>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_check_program_hash(args[0], args[1]))

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
