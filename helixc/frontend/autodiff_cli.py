"""
helixc/frontend/autodiff_cli.py — print the symbolic derivative of a function.

Primary usage:
    python -m helixc.frontend.autodiff_cli <file.hx> <function_name> [<var_name>]
        Prints the symbolic derivative w.r.t. <var_name> (default: first
        parameter). With --as-function, emits a parseable `fn name__grad(...)`
        wrapper.

Introspection (Stage 28.9 + Stage 58 + Stage 59 polish):
    --dump-ast-hashes <file.hx>
        Print `<fn_name> : <12-char hex hash>` for every fn.
    --program-hash <file.hx>
        Print the whole-program structural hash (64 hex).
    --program-signature-hash <file.hx>
        ABI-level hash: covers fn signatures + struct defs, NOT bodies.
    --hash-dump <file.hx>
        Comprehensive JSON dump of all hashes (program / fns / structs
        / modules / signatures) for CI artifact diff-comparison.
    --diff-hash-dump <a.hx> <b.hx>
        Granular per-item drift report (added/removed/changed body/sig).
    --hash-dump-short <file.hx>
        Same as --hash-dump but with 12-hex short hashes (compact logs).
    --diff-trace <a.json> <b.json>
        Diff two trace_to_canonical_json dumps; prints first divergence.
    --diff-program-hash <a.hx> <b.hx>
        Compare two programs: prints SAME or DIFFER + per-fn breakdown.
    --changed-fns <a.hx> <b.hx>
        List fns whose body hash differs between two files.
    --fn-sig-hash <file.hx> <fn_name>
        Print the signature-only hash (ABI-affecting fields only).
    --list-fns <file.hx>
        Enumerate all fns with sig + body hash columns.
    --check-program-hash <file.hx> <expected_hex>
        Assertion-style CI gate: exit 0 if matches, 1 if drift.
    --check-program-signature-hash <file.hx> <expected_hex>
        ABI-level CI gate (body-only refactors don't trip it).
    --list-modules <file.hx>
        Enumerate ModBlock/ModuleDecl entries (incl. nested) with hashes.
    --module-hash <file.hx> <module_name>
        Print the module hash (accepts dotted names for nested).
    --pytree-shape <file.hx> <struct_name>
        Print leaf-path / type / diff-classification for a struct.
    --list-pytrees <file.hx>
        Inventory: leaf-count + diff/non-diff summary per struct.
    --autotune-summary <file.hx>
        Print {fn variants=N} for @autotune @kernel fns + total.
    --autotune-budget <file.hx> <max_total>
        CI gate: exit 0 if total variants <= budget, 1 if over.

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


def _diff_hash_dump(path_a: str, path_b: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: diff two source files
    at every hash granularity.

    Output sections (only shown when non-empty):
      `+ added: <name>` for items present in b but not a
      `- removed: <name>` for items present in a but not b
      `~ changed body: <name>` for fns whose body hash differs but
                                 sig hash matches
      `~ changed sig: <name>` for fns whose signature changed
      `~ changed struct: <name>` for structs that differ
      `~ changed module: <name>` for modules that differ

    Exits 0 with `MATCH` if all hashes identical, 1 otherwise.
    Companion to --diff-program-hash (which is single-line) — this
    gives granular per-item drift detail for code review.
    """
    from .ast_hash import program_hash_dump, short_hash
    src_a = _read_source(path_a)
    src_b = _read_source(path_b)
    prog_a = _parse_or_exit(src_a, path_a)
    prog_b = _parse_or_exit(src_b, path_b)
    a = program_hash_dump(prog_a)
    b = program_hash_dump(prog_b)

    if a == b:
        print(f"MATCH {short_hash(a['program_hash'])}")
        return 0

    # Fns
    a_fns, b_fns = a["fns"], b["fns"]
    for name in sorted(set(b_fns) - set(a_fns)):
        print(f"+ added fn: {name}")
    for name in sorted(set(a_fns) - set(b_fns)):
        print(f"- removed fn: {name}")
    for name in sorted(set(a_fns) & set(b_fns)):
        if a_fns[name] == b_fns[name]:
            continue
        if a_fns[name]["sig_hash"] != b_fns[name]["sig_hash"]:
            print(f"~ changed sig: {name}")
        else:
            print(f"~ changed body: {name}")

    # Structs
    for name in sorted(set(b["structs"]) - set(a["structs"])):
        print(f"+ added struct: {name}")
    for name in sorted(set(a["structs"]) - set(b["structs"])):
        print(f"- removed struct: {name}")
    for name in sorted(set(a["structs"]) & set(b["structs"])):
        if a["structs"][name] != b["structs"][name]:
            print(f"~ changed struct: {name}")

    # Modules
    for name in sorted(set(b["modules"]) - set(a["modules"])):
        print(f"+ added module: {name}")
    for name in sorted(set(a["modules"]) - set(b["modules"])):
        print(f"- removed module: {name}")
    for name in sorted(set(a["modules"]) & set(b["modules"])):
        if a["modules"][name] != b["modules"][name]:
            print(f"~ changed module: {name}")

    return 1


def _hash_dump(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print the comprehensive
    program_hash_dump as pretty-printed JSON.

    Sorted keys + 2-space indent for diff-friendly artifact storage.
    Wraps program_hash_dump() as a script-friendly entry point.

    Use case: emit a 'hash-dump.json' CI artifact, downstream
    gates diff it to detect drift at fn/struct/module granularity
    without per-flag invocations.
    """
    import json
    from .ast_hash import program_hash_dump
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    dump = program_hash_dump(prog)
    print(json.dumps(dump, sort_keys=True, indent=2))
    return 0


def _diff_trace(path_a: str, path_b: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: diff two trace JSON
    dump files (as produced by `trace_to_canonical_json`).

    Reads each path as JSON, deserializes via trace_from_canonical_json,
    and prints:
      `MATCH` (exit 0) if trace_equiv holds, or
      `DIFFER` + the first divergent event (exit 1).

    Use case: golden-trace regression — store a trace JSON dump as
    a CI artifact, then in a later run re-dump and compare to detect
    runtime-behavior drift even when source-code hashes are identical
    (e.g., reproducibility audit catches a nondeterministic op).
    """
    from .trace_pass import trace_from_canonical_json, trace_equiv, trace_diff
    try:
        with open(path_a, "r", encoding="utf-8") as f:
            a_json = f.read()
        with open(path_b, "r", encoding="utf-8") as f:
            b_json = f.read()
    except OSError as e:
        print(f"error: autodiff_cli: {e}", file=sys.stderr)
        return 1
    a = trace_from_canonical_json(a_json)
    b = trace_from_canonical_json(b_json)
    if trace_equiv(a, b):
        print(f"MATCH (events={len(a)})")
        return 0
    print("DIFFER")
    diff = trace_diff(a, b)
    if diff is not None:
        idx, ea, eb = diff
        print(f"  first divergence at event[{idx}]:")
        print(f"    a={ea}")
        print(f"    b={eb}")
    return 1


def _hash_dump_short(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print the compact (12-hex)
    program_hash_dump_short as pretty-printed JSON. Same shape as
    --hash-dump but every hash truncated to 12 hex chars for compact
    log/changelog artifacts.

    Use case: human-readable build-info snippets, daily-rotated
    CI logs where 48-bit collision resistance is sufficient.
    """
    import json
    from .ast_hash import program_hash_dump_short
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    dump = program_hash_dump_short(prog)
    print(json.dumps(dump, sort_keys=True, indent=2))
    return 0


def _print_program_signature_hash(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print the ABI-level
    signature hash of a program. Wraps program_signature_hash() as a
    script-friendly entry point.

    Companion to --program-hash (full structural) and --fn-sig-hash
    (single fn). This one is at the whole-program granularity: same
    hash => ABI-equivalent.

    Use case: CI gate asserting that an internal refactor didn't
    accidentally change the public surface.
    """
    from .ast_hash import program_signature_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    print(program_signature_hash(prog))
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


def _check_program_signature_hash(path: str, expected: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: ABI-level CI gate.
    Exits 0 if the program_signature_hash matches `expected`; 1 if
    drift; 2 if bad args.

    Accepts full 64-hex OR 12-hex short form (prefix match).

    Use case: 'don't accidentally break the public API' gate.
    Pre-commit hook example:
      python -m helixc.frontend.autodiff_cli \\
          --check-program-signature-hash mylib.hx 1760319d55b8
      || { echo 'mylib.hx ABI drifted; bump version'; exit 1; }

    Body-only refactors don't trip the gate (program_signature_hash
    is invariant to body changes); only true API-surface changes do.
    """
    from .ast_hash import program_signature_hash, short_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    actual = program_signature_hash(prog)
    if actual == expected:
        return 0
    if len(expected) == 12 and short_hash(actual) == expected:
        return 0
    print(f"signature hash mismatch")
    print(f"  expected: {expected}")
    print(f"  actual:   {short_hash(actual)} ({actual})")
    return 1


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


def _list_modules(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate ModBlock
    (and ModuleDecl) entries in a source file with their content
    hashes.

    Output format per module (one line):
      `<name> hash=<12hex>`

    For nested ModBlocks, also includes inner modules recursively
    with dotted names (e.g., `outer.inner hash=...`).

    Use case:
    - Inventory the modules in a multi-module file
    - Pre-compute hash table for incremental rebuilds at module
      granularity
    - Detect when a specific submodule changed between commits

    Exit 0 always (no failure mode beyond parse error).
    """
    from .ast_hash import module_hash, short_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    rows: list[tuple[str, str]] = []

    def _walk_modules(items, prefix: str = "") -> None:
        for it in items:
            if isinstance(it, A.ModBlock):
                full = f"{prefix}{it.name}" if not prefix else f"{prefix}.{it.name}"
                rows.append((full, short_hash(module_hash(it))))
                # Recurse into nested ModBlocks.
                _walk_modules(it.items, full)
            elif isinstance(it, A.ModuleDecl):
                # Header-syntax `module path::to::name` — dot-join path.
                full = ".".join(it.path)
                if prefix:
                    full = f"{prefix}.{full}"
                rows.append((full, short_hash(module_hash(it))))

    _walk_modules(prog.items)

    # Sort alphabetically by module name for stable output.
    for name, h in sorted(rows, key=lambda r: r[0]):
        print(f"{name} hash={h}")
    return 0


def _list_pytrees(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: enumerate all structs in
    a file with their pytree leaf count + diff-eligibility summary.

    Output format per struct (one line):
      `<name> leaves=<N> diff=<K> non_diff=<M>` (when flatten succeeds)
      `<name> REJECTED <reason>` (when struct has non-diff fields)

    Sorted alphabetically by struct name. Provides an at-a-glance
    inventory of every struct's gradient-eligibility status without
    requiring per-struct --pytree-shape invocations.

    Use case:
    - Repo audit: catch structs that have drifted into non-diff
      (e.g., added an i32 field that breaks pytree).
    - Manifest generation for training-script param discovery.

    Exit 0 always.
    """
    from .pytree import flatten_pytree
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    struct_decls = {it.name: it for it in prog.items
                    if isinstance(it, A.StructDecl)}
    for name in sorted(struct_decls.keys()):
        try:
            leaves = flatten_pytree(struct_decls[name], struct_decls)
            diff = sum(1 for l in leaves if l.is_diff)
            print(f"{name} leaves={len(leaves)} diff={diff} "
                  f"non_diff={len(leaves) - diff}")
        except ValueError as e:
            # Strip trap-id from error for cleaner output.
            reason = str(e).split("(trap")[0].strip()
            print(f"{name} REJECTED {reason}")
    return 0


def _pytree_shape(path: str, struct_name: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: print the pytree shape
    of a struct — one line per leaf showing path + type + diff/non-diff
    classification.

    Output format per leaf (one line):
      `<path> ty=<ty_name> diff=<True|False>`
    Sorted by leaf path. Trailing summary line: `total leaves=N
    diff=K non_diff=M`.

    Use case:
    - Verify a struct is correctly shaped for gradient computation
      (all gradient-eligible leaves marked diff=True).
    - Pre-flight check before calling grad_rev_all on a struct param.
    - Generate a manifest of trainable params for a model serializer.

    Exit 0 on success, 1 if struct_name not found in the source file.
    """
    from .pytree import flatten_pytree
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    struct_decls = {it.name: it for it in prog.items
                    if isinstance(it, A.StructDecl)}
    if struct_name not in struct_decls:
        print(f"error: autodiff_cli: struct {struct_name!r} not found in {path}",
              file=sys.stderr)
        return 1
    try:
        leaves = flatten_pytree(struct_decls[struct_name], struct_decls)
    except ValueError as e:
        # Pytree rejects non-diff or cyclic structs (trap 26002/26003).
        # Surface the error as a CLI diagnostic rather than a traceback.
        print(f"error: autodiff_cli: pytree shape rejected for "
              f"{struct_name!r}: {e}", file=sys.stderr)
        return 1
    leaves_sorted = sorted(leaves, key=lambda l: l.path)
    diff_count = 0
    for leaf in leaves_sorted:
        print(f"{leaf.path} ty={leaf.ty_name} diff={leaf.is_diff}")
        if leaf.is_diff:
            diff_count += 1
    print(f"total leaves={len(leaves)} diff={diff_count} "
          f"non_diff={len(leaves) - diff_count}")
    return 0


def _autotune_summary(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #8 polish: print the autotune
    variant-count summary for a source file.

    Output format per fn (one line):
      `<fn_name> variants=<count>`
    Sorted alphabetically by fn name. Total line at the end.

    Use case:
    - CI guard: assert no single fn explodes past a budget threshold
    - Pre-commit hook: detect when a kernel's variant count drifts
      unexpectedly (e.g., from an accidentally-doubled autotune list)
    - Quick repository inventory of autotune surface area

    Exit 0 always (no failure mode beyond parse error).
    """
    from .autotune_expand import autotune_expansion_summary
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    summary = autotune_expansion_summary(prog)
    total = 0
    for fn_name in sorted(summary.keys()):
        count = summary[fn_name]
        print(f"{fn_name} variants={count}")
        total += count
    print(f"total variants={total}")
    return 0


def _autotune_budget(path: str, max_total: str) -> int:
    """Stage 59 follow-on / Tier 2 #8 polish: assertion-style budget
    check for total autotune variant count.

    Computes `sum(autotune_expansion_summary(prog).values())` and
    compares to `max_total` (parsed as int). If the total is at or
    below the budget, exits 0 silently. If above, exits 1 with a
    diagnostic showing the per-fn breakdown.

    Exit codes:
      0 — total within budget (silent)
      1 — total exceeds budget (prints breakdown)
      2 — bad arg (parse failure, non-int budget, etc.)

    Use case: CI gate. A repo-wide pre-commit hook can assert
    `--autotune-budget kernels.hx 256` so accidental autotune-list
    blow-ups can't merge silently.
    """
    try:
        budget = int(max_total)
    except ValueError:
        print(f"error: autotune_cli: max_total {max_total!r} is not an int",
              file=sys.stderr)
        return 2
    from .autotune_expand import autotune_expansion_summary
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    summary = autotune_expansion_summary(prog)
    total = sum(summary.values())
    if total <= budget:
        return 0
    print(f"autotune variant count {total} exceeds budget {budget}")
    for fn_name in sorted(summary.keys()):
        print(f"  {fn_name} variants={summary[fn_name]}")
    return 1


def _module_hash_cli(path: str, mod_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print the module_hash
    of a specific module by name. For nested modules, accept dotted
    names (e.g., `outer.inner`).

    Returns exit 0 on success, 1 if the module name is not found in
    the source file.
    """
    from .ast_hash import module_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _find(items, target: str, prefix: str = ""):
        for it in items:
            if isinstance(it, A.ModBlock):
                full = f"{prefix}{it.name}" if not prefix else f"{prefix}.{it.name}"
                if full == target:
                    return it
                found = _find(it.items, target, full)
                if found is not None:
                    return found
            elif isinstance(it, A.ModuleDecl):
                full = ".".join(it.path)
                if prefix:
                    full = f"{prefix}.{full}"
                if full == target:
                    return it
        return None

    mod = _find(prog.items, mod_name)
    if mod is None:
        print(f"error: autodiff_cli: module {mod_name!r} not found in {path}",
              file=sys.stderr)
        return 1
    print(module_hash(mod))
    return 0


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

    Stage 59 enhancement: when the two programs differ at the full
    hash but match at the signature hash, the output adds a
    `kind=body-only` marker — useful for code review (the diff is
    semantic but doesn't break the ABI). When signature hashes also
    differ, `kind=signature-change` is shown.

    Use case: CI sanity check that a refactor PR is semantically
    identical to the baseline (e.g., formatter-only diff, var
    rename, etc.). Build cache: if hashes match, skip recompile.
    """
    from .ast_hash import program_hash, program_signature_hash, short_hash
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
    sig_a = program_signature_hash(prog_a)
    sig_b = program_signature_hash(prog_b)
    if sig_a == sig_b:
        print(f"  kind=body-only (signatures match: {short_hash(sig_a)})")
    else:
        print(f"  kind=signature-change "
              f"(a_sig={short_hash(sig_a)} b_sig={short_hash(sig_b)})")
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

    if "--program-signature-hash" in flags:
        if len(args) < 1:
            print("usage: --program-signature-hash <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_print_program_signature_hash(args[0]))

    if "--hash-dump" in flags:
        if len(args) < 1:
            print("usage: --hash-dump <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_hash_dump(args[0]))

    if "--diff-hash-dump" in flags:
        if len(args) < 2:
            print("usage: --diff-hash-dump <a.hx> <b.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_diff_hash_dump(args[0], args[1]))

    if "--hash-dump-short" in flags:
        if len(args) < 1:
            print("usage: --hash-dump-short <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_hash_dump_short(args[0]))

    if "--diff-trace" in flags:
        if len(args) < 2:
            print("usage: --diff-trace <a.json> <b.json>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_diff_trace(args[0], args[1]))

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

    if "--check-program-signature-hash" in flags:
        if len(args) < 2:
            print("usage: --check-program-signature-hash <file.hx> "
                  "<expected_hex_hash>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_check_program_signature_hash(args[0], args[1]))

    if "--list-modules" in flags:
        if len(args) < 1:
            print("usage: --list-modules <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_modules(args[0]))

    if "--module-hash" in flags:
        if len(args) < 2:
            print("usage: --module-hash <file.hx> <module_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_module_hash_cli(args[0], args[1]))

    if "--autotune-summary" in flags:
        if len(args) < 1:
            print("usage: --autotune-summary <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_autotune_summary(args[0]))

    if "--autotune-budget" in flags:
        if len(args) < 2:
            print("usage: --autotune-budget <file.hx> <max_total>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_autotune_budget(args[0], args[1]))

    if "--pytree-shape" in flags:
        if len(args) < 2:
            print("usage: --pytree-shape <file.hx> <struct_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_pytree_shape(args[0], args[1]))

    if "--list-pytrees" in flags:
        if len(args) < 1:
            print("usage: --list-pytrees <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_pytrees(args[0]))

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
