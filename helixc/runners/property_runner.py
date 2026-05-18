"""Stage 86 — Stage 77 Inc 2: @property test runner.

Discovers all `@property` fns from a Helix program (parsed with the
stdlib loaded) and runs each with a fixed table of representative
inputs. For each (property, input) pair, generates a synthetic
`main` that calls the property and returns 42 on pass / 99 on fail,
compiles via the standard codegen pipeline, runs via WSL, and
aggregates pass/fail counts.

USAGE
=====

    python -m helixc.runners.property_runner \\
        --file helixc/examples/dogfood_23_property_proofs.hx

    python -m helixc.runners.property_runner \\
        --stdlib-only                        # just run safety.hx properties

Output is a pass/fail summary per property + total. Exit code is 0
if all properties pass for all inputs, 1 otherwise. Suitable for
CI integration.

This is the runtime half of the @property scaffolding shipped in
Stage 77 Inc 1: pre-Stage-86, registered @property fns were just
metadata for an external runner. Stage 86 IS that runner.

LIMITATIONS (Phase-0)
=====================

- Only properties of signature `fn name(x: T) -> bool` are runnable
  (single arg). Multi-arg properties are listed but skipped with a
  note. Inc 3 plan: cartesian product across multiple args.
- Input tables are fixed per type (no randomization yet). Inc 4
  plan: pluggable input generators (Hypothesis-style).
- Properties returning false for ANY input fail the whole property.
  No shrinking on failure yet. Inc 5 plan: shrinking.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse  # noqa: E402
from helixc.frontend.typecheck import TypeChecker  # noqa: E402


# Phase-0 fixed input tables per primitive type. Inc 4 will make these
# pluggable via a generator-registry pattern.
_INPUT_TABLE = {
    "i32":  [-1_000_000, -100, -1, 0, 1, 100, 1_000_000],
    "i64":  [-1_000_000_000, -100, -1, 0, 1, 100, 1_000_000_000],
    "u32":  [0, 1, 100, 1_000_000, 4_000_000_000],
    "u64":  [0, 1, 100, 1_000_000_000, 9_000_000_000_000_000_000],
    "f32":  [-1.0e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1.0e6],
    "f64":  [-1.0e9, -1.0, 0.0, 1.0, 1.0e9],
    "bool": [True, False],
}


def _format_input_lit(value, ty: str) -> str:
    """Format a Python value as a Helix literal for the given type."""
    if ty == "bool":
        return "true" if value else "false"
    if ty in ("f32", "f64"):
        # Helix accepts decimal-form floats with a suffix.
        return f"{value!r}_{ty}"
    # Integer types.
    return f"{value}_{ty}"


def _generate_runner_main(prop_name: str, arg_ty: str, value) -> str:
    """Generate a Helix main() that calls prop_name(value_lit) and
    returns 42 iff the property holds for this input, 99 otherwise.
    """
    lit = _format_input_lit(value, arg_ty)
    return textwrap.dedent(f"""
    fn main() -> i32 {{
        if {prop_name}({lit}) {{ 42 }} else {{ 99 }}
    }}
    """).strip() + "\n"


def _discover_properties(prog) -> list[tuple[str, list[tuple[str, str]]]]:
    """Walk prog.items and return a list of (fn_name, [(arg_name,
    arg_ty_name), ...]) for every fn carrying `@property`. We re-
    read the AST (not _property_fn_names) so we keep the arg types
    that the registry discards."""
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for item in prog.items:
        if not hasattr(item, "attrs") or "property" not in item.attrs:
            continue
        if not hasattr(item, "params"):
            continue
        # Each param is FnParam(name, ty). We only need a string
        # rendering of the type's name for input-table lookup.
        arg_pairs = []
        for p in item.params:
            ty_str = _stringify_ty(p.ty)
            arg_pairs.append((p.name, ty_str))
        out.append((item.name, arg_pairs))
    return out


def _stringify_ty(ty) -> str:
    """Best-effort: get the primitive name out of an AST type node.
    Returns the type's base name (e.g., "i32", "f32") or "?" for
    non-primitive types."""
    # AST type nodes have varying shapes; the most common ones for
    # property fn args are TyName (e.g. "i32") and TyGeneric.
    if hasattr(ty, "name") and isinstance(ty.name, str):
        return ty.name
    if hasattr(ty, "base") and isinstance(ty.base, str):
        return ty.base
    return "?"


def run_properties(src: str, verbose: bool = True) -> tuple[int, int, list[str]]:
    """Discover + run all @property fns in `src`.

    Returns (pass_count, fail_count, fail_log) where fail_log is a
    list of human-readable failure strings.
    """
    # Lazy-import compile_and_run so this module is usable for
    # discovery-only flows on systems without WSL set up.
    from helixc.tests.test_codegen import compile_and_run  # noqa: E402

    prog = parse(src, include_stdlib=True)
    tc = TypeChecker(prog)
    errors = tc.check()
    if errors:
        raise RuntimeError(
            f"property-runner: typecheck errors in input:\n"
            + "\n".join(str(e) for e in errors[:10]))
    properties = _discover_properties(prog)
    if verbose:
        print(f"property-runner: discovered {len(properties)} "
              f"@property fns")
    pass_count = 0
    fail_count = 0
    fail_log: list[str] = []
    for prop_name, arg_pairs in properties:
        if len(arg_pairs) != 1:
            if verbose:
                print(f"  SKIP {prop_name}: {len(arg_pairs)}-arg "
                      f"property not supported in Phase-0 (Inc 3 "
                      f"plan: cartesian product)")
            continue
        arg_name, arg_ty = arg_pairs[0]
        inputs = _INPUT_TABLE.get(arg_ty)
        if inputs is None:
            if verbose:
                print(f"  SKIP {prop_name}: no input table for "
                      f"arg type {arg_ty!r}")
            continue
        for value in inputs:
            # Build a fresh source: the original program + a synthetic
            # main. Strip any pre-existing main from src to avoid
            # duplicate-fn errors.
            src_no_main = _strip_main(src)
            runner_src = src_no_main + "\n" + _generate_runner_main(
                prop_name, arg_ty, value)
            try:
                code = compile_and_run(runner_src)
            except Exception as exc:
                fail_count += 1
                msg = (f"{prop_name}({_format_input_lit(value, arg_ty)}) "
                       f": EXC {type(exc).__name__}: {exc}")
                fail_log.append(msg)
                if verbose:
                    print(f"  FAIL {msg}")
                continue
            if code == 42:
                pass_count += 1
                if verbose:
                    print(f"  pass {prop_name}"
                          f"({_format_input_lit(value, arg_ty)})")
            else:
                fail_count += 1
                msg = (f"{prop_name}({_format_input_lit(value, arg_ty)}) "
                       f": exit={code}")
                fail_log.append(msg)
                if verbose:
                    print(f"  FAIL {msg}")
    return pass_count, fail_count, fail_log


def _strip_main(src: str) -> str:
    """Remove any top-level `fn main(...) { ... }` from src so the
    runner can substitute its own synthetic main. Very naïve: looks
    for `fn main` and removes from there to the matching closing
    brace. Skips if no main is present (e.g. stdlib-only runs).
    """
    idx = src.find("fn main")
    if idx < 0:
        return src
    # Find the opening brace after `fn main`.
    brace_open = src.find("{", idx)
    if brace_open < 0:
        return src
    # Walk forward counting braces.
    depth = 0
    i = brace_open
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[:idx] + src[i + 1:]
        i += 1
    return src  # malformed; leave alone


def _build_stdlib_only_src() -> str:
    """Return a minimal source string that triggers stdlib parse.
    The stdlib is auto-included, so any fn body suffices."""
    return "fn _placeholder() -> i32 { 0 }\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Helix @property test runner (Stage 86 / Stage "
                    "77 Inc 2). Discovers @property fns and runs them "
                    "against a fixed input table.")
    ap.add_argument("--file", "-f",
                    help="Helix source file to discover @property fns from. "
                         "Stdlib is auto-included (so safety.hx properties "
                         "always show up).")
    ap.add_argument("--stdlib-only", action="store_true",
                    help="Run only the @property fns shipped in the stdlib "
                         "(safety.hx). Equivalent to passing an empty file.")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Suppress per-input progress output (still prints "
                         "final summary).")
    args = ap.parse_args(argv)
    if not args.file and not args.stdlib_only:
        ap.error("specify --file or --stdlib-only")
    if args.stdlib_only:
        src = _build_stdlib_only_src()
    else:
        with open(args.file, "r", encoding="utf-8") as fh:
            src = fh.read()
    verbose = not args.quiet
    pass_count, fail_count, fail_log = run_properties(src, verbose=verbose)
    total = pass_count + fail_count
    print("")
    print(f"=== property-runner summary ===")
    print(f"total assertions:  {total}")
    print(f"passed:            {pass_count}")
    print(f"failed:            {fail_count}")
    if fail_log:
        print(f"first 10 failures:")
        for msg in fail_log[:10]:
            print(f"  {msg}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
