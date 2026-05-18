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
    --ast-stats <file.hx>
        High-level program stats: fn/struct/module/attr counts.
    --ast-stats-json <file.hx>
        Same as --ast-stats but machine-readable JSON output.
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
    --trace-dump-summary <file.json>
        High-level stats of a trace dump: counts, balance, short hash.
    --trace-dump-summary-json <file.json>
        Same as --trace-dump-summary but machine-readable JSON output.
    --diff-program-hash <a.hx> <b.hx>
        Compare two programs: prints SAME or DIFFER + per-fn breakdown.
    --changed-fns <a.hx> <b.hx>
        List fns whose body hash differs between two files.
    --fn-sig-hash <file.hx> <fn_name>
        Print the signature-only hash (ABI-affecting fields only).
    --list-fns <file.hx>
        Enumerate all fns with sig + body hash columns.
    --list-fns-json <file.hx>
        Same as --list-fns but machine-readable JSON (full 64-hex hashes).
    --list-structs <file.hx>
        Enumerate all structs with field count + content hash.
    --list-structs-json <file.hx>
        Same as --list-structs but machine-readable JSON output.
    --list-modules-json <file.hx>
        Same as --list-modules but machine-readable JSON output.
    --parse-only <file.hx>
        Lightest CI gate: exit 0 if source parses cleanly, 1 on error.
    --list-fn-attrs <file.hx>
        Enumerate all fns with their attribute list (@pure, @trace, etc).
    --fn-callgraph <file.hx> <fn_name>
        Print fn names directly called from inside <fn_name>'s body.
    --fn-callers <file.hx> <fn_name>
        Inverse: print fns that DIRECTLY call <fn_name> (refactor planning).
    --fn-callgraph-all <file.hx>
        Whole-program callgraph as JSON {fn: [callees...]} for tooling.
    --fn-callers-all <file.hx>
        Whole-program INVERSE callgraph as JSON {fn: [callers...]}.
    --fn-reachable-from <file.hx> <entry_fn>
        Transitive closure: BFS over callgraph from entry_fn (dead-code).
    --fn-reachable-to <file.hx> <target_fn>
        Inverse transitive closure: who can reach target_fn (impact zone).
    --fn-call-stats <file.hx>
        Per-fn fan-in / fan-out as JSON for hotspot identification.
    --fn-leaves <file.hx>
        List 'leaf' fns (those that call no other fn) — sorted, one per line.
    --fn-roots <file.hx>
        List 'root' fns (never called locally) — dead-code candidates.
    --fn-recursive <file.hx>
        List fns that DIRECTLY recurse (call themselves in their body).
    --fn-cycles <file.hx>
        Detect mutual-recursion cycles via Tarjan SCC (size >= 2).
    --list-fn-attrs-json <file.hx>
        Same as --list-fn-attrs but machine-readable JSON output.
    --list-fns-by-attr <file.hx> <attr>
        List fns carrying a specific attribute (e.g., 'pure', 'kernel').
    --check-program-hash <file.hx> <expected_hex>
        Assertion-style CI gate: exit 0 if matches, 1 if drift.
    --check-program-hash-from-file <file.hx> <expected_hash_file>
        Same as --check-program-hash but reads expected hash from a file.
    --check-program-signature-hash <file.hx> <expected_hex>
        ABI-level CI gate (body-only refactors don't trip it).
    --check-program-signature-hash-from-file <file.hx> <pin_file>
        Same as --check-program-signature-hash but reads pin from file.
    --list-modules <file.hx>
        Enumerate ModBlock/ModuleDecl entries (incl. nested) with hashes.
    --module-hash <file.hx> <module_name>
        Print the module hash (accepts dotted names for nested).
    --pytree-shape <file.hx> <struct_name>
        Print leaf-path / type / diff-classification for a struct.
    --list-pytrees <file.hx>
        Inventory: leaf-count + diff/non-diff summary per struct.
    --list-pytrees-json <file.hx>
        Same as --list-pytrees but machine-readable JSON output.
    --pytree-leaf-paths <file.hx> <struct_name>
        Print just the leaf paths (one per line, sorted) for scripting.
    --validate-pytrees <file.hx>
        CI gate: validate every struct as a pytree; exit 1 if any fail.
    --validate-pytrees-json <file.hx>
        Same as --validate-pytrees but machine-readable JSON output.
    --autotune-summary <file.hx>
        Print {fn variants=N} for @autotune @kernel fns + total.
    --autotune-summary-json <file.hx>
        Same as --autotune-summary but machine-readable JSON output.
    --validate-autotune <file.hx>
        CI gate: validate every @autotune attr; exit 1 if any malformed.
    --validate-autotune-json <file.hx>
        Same as --validate-autotune but machine-readable JSON output.
    --validate-trace-attrs <file.hx>
        CI gate: validate @trace usage; rejects @trace on extern fns.
    --list-traced-fns <file.hx>
        Enumerate fns carrying @trace (one per line, sorted).
    --list-traced-fns-json <file.hx>
        Same as --list-traced-fns but machine-readable JSON output.
    --validate-all <file.hx>
        Run all 3 validators (pytrees + autotune + trace-attrs) in one shot.
    --validate-all-json <file.hx>
        Same as --validate-all but machine-readable JSON output.
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


def _ast_stats_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --ast-stats in
    machine-readable JSON.

    Output schema:
      {
        "fns": N, "structs": N, "modules": N,
        "autotune_fns": N, "kernel_fns": N, "pure_fns": N,
        "traced_fns": N, "total_attrs": N
      }
    """
    import json
    from .ast_walker import iter_fn_decls
    from .autotune import has_autotune, has_kernel
    from .trace_pass import is_traced
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    fns = list(iter_fn_decls(prog))
    structs = [it for it in prog.items if isinstance(it, A.StructDecl)]
    modules = [it for it in prog.items if isinstance(it, A.ModBlock)]
    result = {
        "fns": len(fns),
        "structs": len(structs),
        "modules": len(modules),
        "autotune_fns": sum(1 for f in fns if has_autotune(f)),
        "kernel_fns": sum(1 for f in fns if has_kernel(f)),
        "pure_fns": sum(1 for f in fns if "pure" in f.attrs),
        "traced_fns": sum(1 for f in fns if is_traced(f)),
        "total_attrs": sum(len(f.attrs) for f in fns),
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _ast_stats(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: high-level program
    structural stats — useful for repo-wide complexity inventories.

    Output (one per line):
      fns=N
      structs=N
      modules=N
      autotune_fns=N (subset of fns with @autotune)
      kernel_fns=N (subset of fns with @kernel)
      pure_fns=N (subset of fns with @pure)
      traced_fns=N (subset of fns with @trace)
      total_attrs=N (sum of attr-counts across all fns)

    Use case:
    - Repo-wide complexity tracking (stat-by-stat across releases)
    - Identify which files have grown disproportionately
    - Sanity-check expected counts after a refactor
    """
    from .ast_walker import iter_fn_decls
    from .autotune import has_autotune, has_kernel
    from .trace_pass import is_traced
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    fns = list(iter_fn_decls(prog))
    structs = [it for it in prog.items if isinstance(it, A.StructDecl)]
    modules = [it for it in prog.items if isinstance(it, A.ModBlock)]
    autotune_count = sum(1 for f in fns if has_autotune(f))
    kernel_count = sum(1 for f in fns if has_kernel(f))
    pure_count = sum(1 for f in fns if "pure" in f.attrs)
    traced_count = sum(1 for f in fns if is_traced(f))
    total_attrs = sum(len(f.attrs) for f in fns)
    print(f"fns={len(fns)}")
    print(f"structs={len(structs)}")
    print(f"modules={len(modules)}")
    print(f"autotune_fns={autotune_count}")
    print(f"kernel_fns={kernel_count}")
    print(f"pure_fns={pure_count}")
    print(f"traced_fns={traced_count}")
    print(f"total_attrs={total_attrs}")
    return 0


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


def _trace_dump_summary_json(path: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: --trace-dump-summary
    in machine-readable JSON form.

    Output schema:
      {
        "events": N,
        "fn_counts": {fn: count, ...},
        "op_kind_counts": {kind: count, ...},
        "balanced": bool,
        "hash_short": "<12hex>",
        "hash_full": "<64hex>"
      }
    """
    import json
    from .trace_pass import (
        trace_from_canonical_json, trace_fn_counts, trace_op_counts,
        trace_is_balanced, trace_hash,
    )
    from .ast_hash import short_hash
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
    except OSError as e:
        print(f"error: autodiff_cli: {e}", file=sys.stderr)
        return 1
    buf = trace_from_canonical_json(s)
    h = trace_hash(buf)
    result = {
        "events": len(buf),
        "fn_counts": dict(sorted(trace_fn_counts(buf).items())),
        "op_kind_counts": dict(sorted(trace_op_counts(buf).items())),
        "balanced": trace_is_balanced(buf),
        "hash_short": short_hash(h),
        "hash_full": h,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _trace_dump_summary(path: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: print high-level summary
    stats of a trace JSON dump.

    Reads a trace_to_canonical_json file and prints:
      `events=N`
      `fn_counts={fn: count, ...}` (sorted by fn name)
      `op_kind_counts={kind: count, ...}` (sorted)
      `balanced=True|False` (entry count == exit count)
      `hash=<12hex>` (short trace_hash for stability check)

    Use case: glance at the high-level shape of a trace without
    reading every event.
    """
    from .trace_pass import (
        trace_from_canonical_json, trace_fn_counts, trace_op_counts,
        trace_is_balanced, trace_hash,
    )
    from .ast_hash import short_hash
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
    except OSError as e:
        print(f"error: autodiff_cli: {e}", file=sys.stderr)
        return 1
    buf = trace_from_canonical_json(s)
    print(f"events={len(buf)}")
    print(f"fn_counts={dict(sorted(trace_fn_counts(buf).items()))}")
    print(f"op_kind_counts={dict(sorted(trace_op_counts(buf).items()))}")
    print(f"balanced={trace_is_balanced(buf)}")
    print(f"hash={short_hash(trace_hash(buf))}")
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


def _check_program_signature_hash_from_file(path: str, expected_file: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --check-program-signature-hash
    variant that reads the expected hash from a pinned-artifact file.

    Symmetric with --check-program-hash-from-file. Reads `expected_file`
    as plain text; the first non-empty stripped line is the expected
    hash. Supports either full 64-hex or 12-hex short form.

    Use case: CI config stores the pinned signature hash as a project
    artifact (.helix-sig-hash) committed alongside the source — for
    'don't break the public API' gates that survive internal refactors.

    Exit codes:
      0 — match
      1 — mismatch (prints expected vs actual)
      2 — bad arg (parse failure, missing expected_file, etc.)
    """
    try:
        with open(expected_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except OSError as e:
        print(f"error: autodiff_cli: {e}", file=sys.stderr)
        return 2
    if not lines:
        print(f"error: autodiff_cli: {expected_file!r} contains no hash",
              file=sys.stderr)
        return 2
    return _check_program_signature_hash(path, lines[0])


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


def _check_program_hash_from_file(path: str, expected_file: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --check-program-hash
    variant that reads the expected hash from a pinned-artifact file
    rather than a command-line argument.

    Reads `expected_file` as plain text; the first non-empty stripped
    line is the expected hash. Supports either the full 64-hex or
    the 12-hex short form (prefix match).

    Use case: CI config stores the pinned hash as a project artifact
    (.helix-hash) committed alongside the source. The pre-commit hook
    becomes:
      python -m helixc.frontend.autodiff_cli \\
          --check-program-hash-from-file stdlib/result.hx .helix-hash \\
        || { echo 'stdlib/result.hx drifted from pin'; exit 1; }

    Cleaner than embedding the 64-hex in a shell script — file is
    version-controlled, single source of truth.

    Exit codes:
      0 — match
      1 — mismatch (prints expected vs actual)
      2 — bad arg (parse failure, missing expected_file, etc.)
    """
    try:
        with open(expected_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except OSError as e:
        print(f"error: autodiff_cli: {e}", file=sys.stderr)
        return 2
    if not lines:
        print(f"error: autodiff_cli: {expected_file!r} contains no hash",
              file=sys.stderr)
        return 2
    return _check_program_hash(path, lines[0])


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


def _validate_pytrees_json(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: --validate-pytrees in
    machine-readable JSON form.

    Output schema:
      {
        "structs": {
          "<name>": {"status": "OK"|"FAIL", "diags": [...]}
        },
        "total": {"structs": N, "ok": K, "fail": M}
      }
    """
    import json
    from .pytree import validate_pytree
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    struct_decls = {it.name: it for it in prog.items
                    if isinstance(it, A.StructDecl)}
    structs: dict = {}
    ok_count = 0
    fail_count = 0
    for name in sorted(struct_decls.keys()):
        diags = validate_pytree(struct_decls[name], struct_decls)
        if not diags:
            structs[name] = {"status": "OK", "diags": []}
            ok_count += 1
        else:
            structs[name] = {"status": "FAIL", "diags": diags}
            fail_count += 1
    result = {
        "structs": structs,
        "total": {
            "structs": ok_count + fail_count,
            "ok": ok_count,
            "fail": fail_count,
        },
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 1 if fail_count else 0


def _validate_pytrees(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: validate every struct
    in a file as a pytree.

    Output:
      `OK <name>` for each struct that flattens successfully
      `FAIL <name>: <diagnostic>` for each that doesn't

    Trailing line: `total structs=N OK=K FAIL=M`.

    Exit 0 if every struct OK, 1 if any FAIL, regardless of count.
    Use case: CI gate — assert no struct in the file has drifted
    into a non-flattenable shape.
    """
    from .pytree import validate_pytree
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    struct_decls = {it.name: it for it in prog.items
                    if isinstance(it, A.StructDecl)}
    ok_count = 0
    fail_count = 0
    for name in sorted(struct_decls.keys()):
        diags = validate_pytree(struct_decls[name], struct_decls)
        if not diags:
            print(f"OK {name}")
            ok_count += 1
        else:
            for diag in diags:
                # Strip trap-id parenthesis for cleaner output.
                clean = diag.split("(trap")[0].strip()
                print(f"FAIL {name}: {clean}")
            fail_count += 1
    total = ok_count + fail_count
    print(f"total structs={total} OK={ok_count} FAIL={fail_count}")
    return 1 if fail_count else 0


def _pytree_leaf_paths(path: str, struct_name: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: print just the leaf paths
    of a struct's pytree, one per line — no types, no diff flag.

    Lighter alternative to --pytree-shape when scripts want JUST the
    paths (e.g., piping through xargs to drive per-leaf operations).

    Sorted alphabetically for deterministic ordering. Exit 0 success,
    1 if struct not found OR pytree-rejected.
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
        print(f"error: autodiff_cli: pytree shape rejected for "
              f"{struct_name!r}: {e}", file=sys.stderr)
        return 1
    for leaf in sorted(leaves, key=lambda l: l.path):
        print(leaf.path)
    return 0


def _list_pytrees_json(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: --list-pytrees in
    machine-readable JSON form.

    Output schema:
      {
        "<struct_name>": {
          "status": "OK" | "REJECTED",
          "leaves": N, "diff": K, "non_diff": M,  (OK case)
          OR
          "reason": "<diagnostic>",               (REJECTED case)
        }
      }
    """
    import json
    from .pytree import flatten_pytree
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    struct_decls = {it.name: it for it in prog.items
                    if isinstance(it, A.StructDecl)}
    result: dict = {}
    for name in sorted(struct_decls.keys()):
        try:
            leaves = flatten_pytree(struct_decls[name], struct_decls)
            diff = sum(1 for l in leaves if l.is_diff)
            result[name] = {
                "status": "OK",
                "leaves": len(leaves),
                "diff": diff,
                "non_diff": len(leaves) - diff,
            }
        except ValueError as e:
            result[name] = {
                "status": "REJECTED",
                "reason": str(e).split("(trap")[0].strip(),
            }
    print(json.dumps(result, sort_keys=True, indent=2))
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


def _validate_all_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --validate-all output
    as machine-readable JSON.

    Output schema:
      {
        "pytrees":     {"status": "OK"|"FAIL", "diags": [...]},
        "autotune":    {"status": "OK"|"FAIL", "diags": [...]},
        "trace-attrs": {"status": "OK"|"FAIL", "diags": [...]},
        "total": {"validators": 3, "ok": K, "fail": M}
      }

    Exit 0 if all pass, 1 if any fails — same as --validate-all but
    JSON output for CI integration (downstream tools can parse the
    structured result instead of regex-matching the human format).
    """
    import json
    from .pytree import validate_pytree
    from .autotune import validate_autotune_prog
    from .trace_pass import validate_trace_attrs

    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    struct_decls = {it.name: it for it in prog.items
                    if isinstance(it, A.StructDecl)}
    pytree_diags: list[str] = []
    for name in sorted(struct_decls.keys()):
        for msg in validate_pytree(struct_decls[name], struct_decls):
            pytree_diags.append(f"{name}: {msg}")

    autotune_diags = validate_autotune_prog(prog)
    trace_diags = validate_trace_attrs(prog)

    def _entry(diags: list[str]) -> dict:
        return {
            "status": "OK" if not diags else "FAIL",
            "diags": diags,
        }

    result = {
        "pytrees": _entry(pytree_diags),
        "autotune": _entry(autotune_diags),
        "trace-attrs": _entry(trace_diags),
    }
    fail_count = sum(1 for e in result.values() if e["status"] == "FAIL")
    result["total"] = {
        "validators": 3,
        "ok": 3 - fail_count,
        "fail": fail_count,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 1 if fail_count else 0


def _validate_all(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: run every Phase-0
    validator in one shot.

    Aggregates:
      - validate_pytree (every struct)
      - validate_autotune_prog (every @autotune fn)
      - validate_trace_attrs (every @trace fn)

    Output: per-validator status header + diagnostics if any.
      `[pytrees] OK` or `[pytrees] FAIL (<count>)` + list
      `[autotune] OK` or `[autotune] FAIL (<count>)` + list
      `[trace-attrs] OK` or `[trace-attrs] FAIL (<count>)` + list

    Trailing: `total validators=3 OK=K FAIL=M`.

    Exit 0 if all validators clean, 1 if any fail.

    Use case: one-shot pre-commit gate. Faster than 3 separate CLI
    invocations.
    """
    from .pytree import validate_pytree
    from .autotune import validate_autotune_prog
    from .trace_pass import validate_trace_attrs

    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    struct_decls = {it.name: it for it in prog.items
                    if isinstance(it, A.StructDecl)}
    pytree_diags: list[str] = []
    for name in sorted(struct_decls.keys()):
        d = validate_pytree(struct_decls[name], struct_decls)
        for msg in d:
            pytree_diags.append(f"{name}: {msg}")

    autotune_diags = validate_autotune_prog(prog)
    trace_diags = validate_trace_attrs(prog)

    fail_count = 0

    def _emit(tag: str, diags: list[str]) -> None:
        nonlocal fail_count
        if not diags:
            print(f"[{tag}] OK")
        else:
            print(f"[{tag}] FAIL ({len(diags)})")
            for d in diags:
                print(f"  {d}")
            fail_count += 1

    _emit("pytrees", pytree_diags)
    _emit("autotune", autotune_diags)
    _emit("trace-attrs", trace_diags)

    print(f"total validators=3 OK={3 - fail_count} FAIL={fail_count}")
    return 1 if fail_count else 0


def _list_traced_fns_json(path: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: --list-traced-fns in
    machine-readable JSON form.

    Output: {"traced_fns": ["fn1", "fn2", ...]} sorted alphabetically.
    """
    import json
    from .trace_pass import traced_fn_names
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    print(json.dumps({"traced_fns": sorted(traced_fn_names(prog))},
                       sort_keys=True, indent=2))
    return 0


def _list_traced_fns(path: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: list fns carrying
    @trace, one per line. Sorted alphabetically. Walks ModBlock-
    nested fns via iter_fn_decls (no nested-scope blind spot).

    Use case:
    - Audit which fns are getting trace-logged
    - Verify a refactor preserved @trace coverage
    - Drive a per-fn benchmark harness for traced functions only
    """
    from .trace_pass import traced_fn_names
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for name in sorted(traced_fn_names(prog)):
        print(name)
    return 0


def _validate_trace_attrs(path: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: run validate_trace_attrs
    over a file. Phase-0 rules:
      * @trace on extern \"C\" fn is rejected (no body to instrument)
      * @trace on @pure fn is allowed (tracing is observer-only)

    Exits 0 silently on clean, 1 with diagnostics on rule violation.
    """
    from .trace_pass import validate_trace_attrs
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    diags = validate_trace_attrs(prog)
    if not diags:
        return 0
    for d in diags:
        print(d)
    return 1


def _validate_autotune_json(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #8 polish: --validate-autotune
    in machine-readable JSON form.

    Output schema:
      {
        "diags": [...],
        "total": {"count": N},
        "status": "OK"|"FAIL"
      }
    """
    import json
    from .autotune import validate_autotune_prog
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    diags = validate_autotune_prog(prog)
    result = {
        "diags": diags,
        "total": {"count": len(diags)},
        "status": "OK" if not diags else "FAIL",
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 1 if diags else 0


def _validate_autotune(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #8 polish: run validate_autotune_prog
    over a file and print all diagnostics. Exit 0 if clean, 1 if any.

    Pre-existing validate_autotune_prog (Audit 28.8 A12) was exposed
    via helixc.check; this surfaces it at the CLI for standalone
    autotune-policy assertions.

    Use case: pre-commit hook asserting no @autotune attribute has
    drifted into a malformed state (e.g., empty value list, non-int
    values, oversized Cartesian product per trap 27001).
    """
    from .autotune import validate_autotune_prog
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    diags = validate_autotune_prog(prog)
    if not diags:
        return 0
    for d in diags:
        print(d)
    return 1


def _autotune_summary_json(path: str) -> int:
    """Stage 59 follow-on / Tier 2 #8 polish: --autotune-summary in
    machine-readable JSON form.

    Output schema:
      {
        "fns": {"<fn_name>": <variant_count>, ...},
        "total": <int>
      }
    """
    import json
    from .autotune_expand import autotune_expansion_summary
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    summary = autotune_expansion_summary(prog)
    result = {
        "fns": dict(sorted(summary.items())),
        "total": sum(summary.values()),
    }
    print(json.dumps(result, sort_keys=True, indent=2))
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


def _list_fns_by_attr(path: str, attr: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: list fns carrying a
    specific attribute (e.g., 'pure', 'trace', 'kernel', 'autotune').

    Output: one fn name per line, sorted alphabetically. Walks
    ModBlock-nested fns via iter_fn_decls.

    Use case: targeted attribute audits.
      python -m helixc.frontend.autodiff_cli --list-fns-by-attr foo.hx pure
      python -m helixc.frontend.autodiff_cli --list-fns-by-attr foo.hx kernel
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    matching = sorted(
        f.name for f in iter_fn_decls(prog)
        if attr in f.attrs
    )
    for name in matching:
        print(name)
    return 0


def _list_fn_attrs_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-fn-attrs in
    machine-readable JSON form.

    Output schema:
      {
        "<fn_name>": [<attr1>, <attr2>, ...],   # attrs sorted
        ...
      }

    Use case: tooling that wants to query 'which fns are @pure?'
    or compute attribute coverage stats without parsing the human-
    formatted output.
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    result = {
        fn.name: sorted(fn.attrs)
        for fn in iter_fn_decls(prog)
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _fn_recursive(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: list fns that DIRECTLY
    recurse (call themselves from their own body).

    Direct recursion only — mutual recursion (A→B→A) is detected
    separately by --fn-cycles.

    Use case: identify candidates for tail-call optimization audit,
    or for proof obligations (termination proofs are stricter for
    recursive fns).
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _calls_self(fn) -> bool:
        found = [False]
        target = fn.name

        def _walk(n) -> None:
            if found[0] or n is None:
                return
            if isinstance(n, A.Call):
                if isinstance(n.callee, A.Name) and n.callee.name == target:
                    found[0] = True
                    return
                if (isinstance(n.callee, A.Path)
                        and n.callee.segments
                        and n.callee.segments[-1] == target):
                    found[0] = True
                    return
            if hasattr(n, "__dataclass_fields__"):
                for f in n.__dataclass_fields__:
                    v = getattr(n, f)
                    if isinstance(v, list):
                        for x in v:
                            _walk(x)
                    elif isinstance(v, tuple):
                        for x in v:
                            _walk(x)
                    else:
                        _walk(v)

        if fn.body is not None:
            _walk(fn.body)
        return found[0]

    recursive: list[str] = []
    for fn in iter_fn_decls(prog):
        if _calls_self(fn):
            recursive.append(fn.name)
    for name in sorted(recursive):
        print(name)
    return 0


def _fn_cycles(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: detect cycles
    (strongly-connected components of size ≥ 2) in the local
    callgraph. Direct self-recursion is excluded (use --fn-recursive
    for that); this is for mutual recursion: A→B→A, A→B→C→A, etc.

    Output: one cycle per line, formatted as
      `<fn1> -> <fn2> -> ... -> <fn1>` (with the start fn closing
      the loop back).
    Sorted lexicographically by canonical-rotation (the cycle is
    rotated to start at its alphabetically-smallest member).

    Uses Tarjan's SCC algorithm on the local callgraph.

    Use case: termination-proof obligations (mutual recursion is
    harder to prove), inlining-decision input (cycles can't be
    fully inlined).
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    # Build adjacency: fn_name -> set of callees that are also locally-
    # defined fns (filter out external calls for the cycle detector).
    graph: dict[str, set[str]] = {}
    all_fn_names: set[str] = set()
    for fn in iter_fn_decls(prog):
        all_fn_names.add(fn.name)
    for fn in iter_fn_decls(prog):
        callees: set[str] = set()

        def _walk(n, out: set) -> None:
            if n is None:
                return
            if isinstance(n, A.Call):
                if isinstance(n.callee, A.Name):
                    out.add(n.callee.name)
                elif (isinstance(n.callee, A.Path)
                      and n.callee.segments):
                    out.add(n.callee.segments[-1])
            if hasattr(n, "__dataclass_fields__"):
                for f in n.__dataclass_fields__:
                    v = getattr(n, f)
                    if isinstance(v, list):
                        for x in v:
                            _walk(x, out)
                    elif isinstance(v, tuple):
                        for x in v:
                            _walk(x, out)
                    else:
                        _walk(v, out)

        if fn.body is not None:
            _walk(fn.body, callees)
        # Filter to local fns + drop self-loops (handled by --fn-recursive).
        graph[fn.name] = {c for c in callees
                          if c in all_fn_names and c != fn.name}

    # Tarjan's SCC.
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def _strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, set()):
            if w not in index:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) >= 2:
                sccs.append(scc)

    for v in sorted(graph.keys()):
        if v not in index:
            _strongconnect(v)

    # Canonicalize each cycle: rotate so the alphabetically-smallest
    # member comes first.
    def _canonicalize(scc: list[str]) -> list[str]:
        min_idx = min(range(len(scc)), key=lambda i: scc[i])
        return scc[min_idx:] + scc[:min_idx]

    formatted = sorted(_canonicalize(s) for s in sccs)
    for cycle in formatted:
        # Format as A -> B -> C -> A
        print(" -> ".join(cycle) + " -> " + cycle[0])
    return 0


def _fn_leaves(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: list 'leaf' fns —
    fns that DON'T call any other (locally-defined or otherwise) fn.

    A leaf fn is one whose body contains no A.Call nodes (with a
    resolvable Name/Path callee). One leaf-fn name per line, sorted.

    Use case: bottom-up analysis. Leaf fns are the base of the
    callgraph DAG — natural starting points for proof-carrying terms,
    inlining candidates, or vectorization analysis.

    Phase-0 limitation matches --fn-callgraph: indirect calls aren't
    tracked, so a fn whose only calls go through fn pointers will be
    incorrectly marked as a leaf.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _has_any_call(node) -> bool:
        found = [False]

        def _walk(n) -> None:
            if found[0] or n is None:
                return
            if isinstance(n, A.Call):
                if (isinstance(n.callee, A.Name)
                        or (isinstance(n.callee, A.Path) and n.callee.segments)):
                    found[0] = True
                    return
            if hasattr(n, "__dataclass_fields__"):
                for f in n.__dataclass_fields__:
                    v = getattr(n, f)
                    if isinstance(v, list):
                        for x in v:
                            _walk(x)
                    elif isinstance(v, tuple):
                        for x in v:
                            _walk(x)
                    else:
                        _walk(v)

        _walk(node)
        return found[0]

    leaves: list[str] = []
    for fn in iter_fn_decls(prog):
        if fn.body is None or not _has_any_call(fn.body):
            leaves.append(fn.name)
    for name in sorted(leaves):
        print(name)
    return 0


def _fn_roots(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: list 'root' fns — fns
    that are NEVER called by any other locally-defined fn (i.e., have
    zero in-edges in the local callgraph).

    Use case: dead-code detection candidates (unreachable from any
    entry point), public-API enumeration (the fns no internal code
    calls are likely public surfaces).

    Caveat: a fn never-called locally may still be invoked externally
    (FFI export, dispatched via fn ptr, main entry). False-positive
    rate depends on use case — this is a *candidate* list, not a
    definitive dead-code marker.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fn_names: set[str] = set()
    called_names: set[str] = set()

    def _collect_calls(node) -> None:
        if node is None:
            return
        if isinstance(node, A.Call):
            if isinstance(node.callee, A.Name):
                called_names.add(node.callee.name)
            elif (isinstance(node.callee, A.Path)
                  and node.callee.segments):
                called_names.add(node.callee.segments[-1])
        if hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__:
                v = getattr(node, f)
                if isinstance(v, list):
                    for x in v:
                        _collect_calls(x)
                elif isinstance(v, tuple):
                    for x in v:
                        _collect_calls(x)
                else:
                    _collect_calls(v)

    for fn in iter_fn_decls(prog):
        all_fn_names.add(fn.name)
        if fn.body is not None:
            _collect_calls(fn.body)

    roots = sorted(all_fn_names - called_names)
    for name in roots:
        print(name)
    return 0


def _fn_call_stats(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-fn call-graph
    fan-in / fan-out stats as JSON.

    Output schema:
      {
        "<fn_name>": {
          "fan_in": <int>,    # number of distinct local fns that call this
          "fan_out": <int>,   # number of distinct local fns this calls
        }
      }

    Pure summary derivation — same data underlying --fn-callgraph-all
    and --fn-callers-all, but counted instead of listed.

    Use cases:
    - Hotspot identification: high fan_in fns are critical paths
    - Refactor risk assessment: high fan_out fns touch many places
    - Code-health metrics: track distribution shifts across releases
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    fan_in: dict[str, set] = {fn.name: set() for fn in all_fns}
    fan_out: dict[str, set] = {fn.name: set() for fn in all_fns}

    def _collect(node, from_fn: str) -> None:
        if node is None:
            return
        if isinstance(node, A.Call):
            tgt = None
            if isinstance(node.callee, A.Name):
                tgt = node.callee.name
            elif (isinstance(node.callee, A.Path)
                  and node.callee.segments):
                tgt = node.callee.segments[-1]
            if tgt is not None and tgt in all_names:
                fan_out[from_fn].add(tgt)
                fan_in[tgt].add(from_fn)
        if hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__:
                v = getattr(node, f)
                if isinstance(v, list):
                    for x in v:
                        _collect(x, from_fn)
                elif isinstance(v, tuple):
                    for x in v:
                        _collect(x, from_fn)
                else:
                    _collect(v, from_fn)

    for fn in all_fns:
        if fn.body is not None:
            _collect(fn.body, fn.name)

    result = {
        name: {"fan_in": len(fan_in[name]), "fan_out": len(fan_out[name])}
        for name in all_names
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _fn_reachable_to(path: str, target_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: inverse transitive
    callgraph closure. Print all fns that can transitively reach
    `target_name` via some call chain (BFS over the reversed
    callgraph).

    Output: one fn per line, sorted alphabetically. Includes
    `target_name` itself.

    Inverse of --fn-reachable-from. Pair use cases:
    - 'Who depends on fn X?' transitively (impact zone of a
      signature change at any caller-depth)
    - Find all fns whose behavior changes if X changes
    - Test-coverage planning: any fn in this set's tests must
      exercise X's behavior

    Exit 0 on success, 1 if target_name not found.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    if target_name not in all_names:
        print(f"error: autodiff_cli: fn {target_name!r} not found in {path}",
              file=sys.stderr)
        return 1

    # Build inverse adjacency: target_fn -> set of callers (local only).
    callers: dict[str, set] = {fn.name: set() for fn in all_fns}

    def _collect_calls(node, from_fn: str) -> None:
        if node is None:
            return
        if isinstance(node, A.Call):
            tgt = None
            if isinstance(node.callee, A.Name):
                tgt = node.callee.name
            elif (isinstance(node.callee, A.Path)
                  and node.callee.segments):
                tgt = node.callee.segments[-1]
            if tgt is not None and tgt in all_names:
                callers[tgt].add(from_fn)
        if hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__:
                v = getattr(node, f)
                if isinstance(v, list):
                    for x in v:
                        _collect_calls(x, from_fn)
                elif isinstance(v, tuple):
                    for x in v:
                        _collect_calls(x, from_fn)
                else:
                    _collect_calls(v, from_fn)

    for fn in all_fns:
        if fn.body is not None:
            _collect_calls(fn.body, fn.name)

    # BFS in the reverse direction from target.
    reachable: set = {target_name}
    frontier: list = [target_name]
    while frontier:
        next_frontier: list = []
        for v in frontier:
            for w in callers.get(v, set()):
                if w not in reachable:
                    reachable.add(w)
                    next_frontier.append(w)
        frontier = next_frontier

    for name in sorted(reachable):
        print(name)
    return 0


def _fn_reachable_from(path: str, entry_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: transitive callgraph
    closure from an entry-point fn. BFS over the local callgraph
    starting at `entry_name`.

    Output: one reachable fn name per line (including entry_name
    itself), sorted alphabetically.

    Use cases:
    - Dead-code elimination: 'starting from main, what fns are
      actually used?' — the complement of this set is dead.
    - Tree-shaking analysis: minimal-build computation for a given
      entry point.
    - Module-graph dependency tracing.

    Phase-0 limitation matches the rest of the call-graph sub-arc:
    indirect calls (fn pointers, dispatch) not tracked.

    Exit 0 on success, 1 if entry_name not found.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    if entry_name not in all_names:
        print(f"error: autodiff_cli: fn {entry_name!r} not found in {path}",
              file=sys.stderr)
        return 1

    # Build adjacency: fn_name -> set of callees (locally-defined only).
    graph: dict[str, set] = {}
    for fn in all_fns:
        callees: set = set()

        def _walk(n) -> None:
            if n is None:
                return
            if isinstance(n, A.Call):
                if isinstance(n.callee, A.Name):
                    callees.add(n.callee.name)
                elif (isinstance(n.callee, A.Path)
                      and n.callee.segments):
                    callees.add(n.callee.segments[-1])
            if hasattr(n, "__dataclass_fields__"):
                for f in n.__dataclass_fields__:
                    v = getattr(n, f)
                    if isinstance(v, list):
                        for x in v:
                            _walk(x)
                    elif isinstance(v, tuple):
                        for x in v:
                            _walk(x)
                    else:
                        _walk(v)

        if fn.body is not None:
            _walk(fn.body)
        graph[fn.name] = {c for c in callees if c in all_names}

    # BFS from entry.
    reachable: set = {entry_name}
    frontier: list = [entry_name]
    while frontier:
        next_frontier: list = []
        for v in frontier:
            for w in graph.get(v, set()):
                if w not in reachable:
                    reachable.add(w)
                    next_frontier.append(w)
        frontier = next_frontier

    for name in sorted(reachable):
        print(name)
    return 0


def _fn_callers_all(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: whole-program inverse
    callgraph as JSON. Output:
      {
        "<fn_name>": ["<caller1>", "<caller2>", ...],  # sorted, deduped
        ...
      }

    For every locally-defined fn, lists the fns that DIRECTLY call
    it. Inverse pair to --fn-callgraph-all.

    Use cases:
    - Tooling: 'who calls fn X?' answered for all X in one read
    - Hotspot analysis: high-fan-in fns (many callers) are critical
      paths — refactoring them needs extra care
    - Dead-code analysis: fns with empty caller lists are unreachable
      from any local fn (matches --fn-roots)
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    callers: dict[str, set] = {fn.name: set() for fn in all_fns}

    def _collect_calls(node, from_fn: str) -> None:
        if node is None:
            return
        if isinstance(node, A.Call):
            tgt = None
            if isinstance(node.callee, A.Name):
                tgt = node.callee.name
            elif (isinstance(node.callee, A.Path)
                  and node.callee.segments):
                tgt = node.callee.segments[-1]
            if tgt is not None and tgt in all_names:
                callers[tgt].add(from_fn)
        if hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__:
                v = getattr(node, f)
                if isinstance(v, list):
                    for x in v:
                        _collect_calls(x, from_fn)
                elif isinstance(v, tuple):
                    for x in v:
                        _collect_calls(x, from_fn)
                else:
                    _collect_calls(v, from_fn)

    for fn in all_fns:
        if fn.body is not None:
            _collect_calls(fn.body, fn.name)

    result = {name: sorted(c) for name, c in callers.items()}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _fn_callgraph_all(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: whole-program call
    graph as JSON. Output:
      {
        "<fn_name>": ["<callee1>", "<callee2>", ...],  # sorted, deduped
        ...
      }

    For every fn in the file, list its direct callees. Companion to
    --fn-callgraph (single-fn) and --fn-callers (inverse single-fn).

    Use case: feed into a graph-viz tool, compute strongly-connected
    components, find cycles, identify leaf fns, dead-code analysis.
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    result: dict = {}

    def _collect_callees(node, into: set) -> None:
        if node is None:
            return
        if isinstance(node, A.Call):
            if isinstance(node.callee, A.Name):
                into.add(node.callee.name)
            elif (isinstance(node.callee, A.Path)
                  and node.callee.segments):
                into.add(node.callee.segments[-1])
        if hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__:
                v = getattr(node, f)
                if isinstance(v, list):
                    for x in v:
                        _collect_callees(x, into)
                elif isinstance(v, tuple):
                    for x in v:
                        _collect_callees(x, into)
                else:
                    _collect_callees(v, into)

    for fn in iter_fn_decls(prog):
        callees: set[str] = set()
        if fn.body is not None:
            _collect_callees(fn.body, callees)
        result[fn.name] = sorted(callees)

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _fn_callers(path: str, target_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: inverse call-graph
    introspection. Print fn names that DIRECTLY call `target_name`
    (the inverse of --fn-callgraph).

    Walks every fn's body looking for `A.Call` nodes whose callee
    name matches target_name. One caller per line, sorted alpha,
    de-duplicated.

    Use case: refactor planning. 'Who calls fn X?' — answers the
    impact-zone question when contemplating a rename / signature
    change / deletion.

    Note: target_name itself need not exist in the file (the
    function might be a builtin or external). No error is raised
    for an unknown target — empty output simply means no callers.
    Exit 0 always.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    callers: set[str] = set()

    def _has_target_call(node) -> bool:
        found = [False]

        def _walk(n) -> None:
            if found[0] or n is None:
                return
            if isinstance(n, A.Call):
                if isinstance(n.callee, A.Name) and n.callee.name == target_name:
                    found[0] = True
                    return
                if (isinstance(n.callee, A.Path)
                        and n.callee.segments
                        and n.callee.segments[-1] == target_name):
                    found[0] = True
                    return
            if hasattr(n, "__dataclass_fields__"):
                for f in n.__dataclass_fields__:
                    v = getattr(n, f)
                    if isinstance(v, list):
                        for x in v:
                            _walk(x)
                    elif isinstance(v, tuple):
                        for x in v:
                            _walk(x)
                    else:
                        _walk(v)

        _walk(node)
        return found[0]

    for fn in iter_fn_decls(prog):
        if fn.body is not None and _has_target_call(fn.body):
            callers.add(fn.name)

    for name in sorted(callers):
        print(name)
    return 0


def _fn_callgraph(path: str, fn_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: static call-graph
    introspection for one fn. Print fn names called from inside
    `fn_name`'s body, sorted alphabetically. One per line.

    Walks the body AST collecting `A.Call` expressions whose callee
    resolves to a `Name`. Only direct fn-name calls captured —
    method calls, builtin dispatch, and indirect calls via fn
    pointers are not enumerated (Phase-0 limitation; could be
    extended).

    Use case: refactor planning. 'If I rename fn X, who do I need
    to update?' is the inverse question (callers of X); this one
    is 'what does fn X depend on?' (callees of X). Together they
    bracket the impact zone of a refactor.

    Exit 0 on success, 1 if fn_name not found.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    target = None
    for it in prog.items:
        if isinstance(it, A.FnDecl) and it.name == fn_name:
            target = it
            break
    if target is None:
        print(f"error: autodiff_cli: fn {fn_name!r} not found in {path}",
              file=sys.stderr)
        return 1

    callees: set[str] = set()

    def _walk(node) -> None:
        if node is None:
            return
        if isinstance(node, A.Call):
            if isinstance(node.callee, A.Name):
                callees.add(node.callee.name)
            elif isinstance(node.callee, A.Path):
                # Dotted path; record last segment as callee name.
                if node.callee.segments:
                    callees.add(node.callee.segments[-1])
        # Walk all dataclass fields generically.
        if hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__:
                v = getattr(node, f)
                if isinstance(v, list):
                    for x in v:
                        _walk(x)
                elif isinstance(v, tuple):
                    for x in v:
                        _walk(x)
                else:
                    _walk(v)

    if target.body is not None:
        _walk(target.body)
    for name in sorted(callees):
        print(name)
    return 0


def _list_fn_attrs(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate all fns with
    their attribute list.

    Output format per fn (one line):
      `<fn_name>: <attrs sorted, space-separated, or '(no attrs)'>`

    Sorted alphabetically by fn name. Walks ModBlock-nested fns too
    via iter_fn_decls.

    Use case:
    - Repo audit: which fns are @pure? Which carry @trace?
    - Verify a refactor preserved all attributes
    - Find @inline candidates that aren't tagged
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    fns = sorted(iter_fn_decls(prog), key=lambda f: f.name)
    for fn in fns:
        if fn.attrs:
            attrs_str = " ".join(sorted(fn.attrs))
        else:
            attrs_str = "(no attrs)"
        print(f"{fn.name}: {attrs_str}")
    return 0


def _parse_only(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: parse-only CI gate.

    Reads the source + invokes `parse(src)`. Exits 0 on clean parse;
    1 on any parse error (the diagnostic is already emitted by
    `_parse_or_exit`).

    Lightest possible 'does this file compile?' CI check — skips
    typecheck, struct mono, AD lowering, etc. Useful when the only
    question is whether the file is syntactically valid (e.g.,
    after a mechanical refactor, formatter run, or import-rewrite
    bot).
    """
    src = _read_source(path)
    _parse_or_exit(src, path)
    return 0


def _list_structs_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-structs in
    machine-readable JSON form.

    Output: {struct_name: {fields: N, hash: <64hex>}} sorted.
    """
    import json
    from .ast_hash import structural_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    structs = sorted(
        (it for it in prog.items if isinstance(it, A.StructDecl)),
        key=lambda s: s.name,
    )
    result = {
        s.name: {"fields": len(s.fields), "hash": structural_hash(s)}
        for s in structs
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _list_modules_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-modules in
    machine-readable JSON form.

    Output: {module_name: <64hex>} for all ModBlock/ModuleDecl
    (including nested via dotted names).
    """
    import json
    from .ast_hash import module_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    result: dict = {}

    def _walk(items, prefix: str = "") -> None:
        for it in items:
            if isinstance(it, A.ModBlock):
                full = f"{prefix}{it.name}" if not prefix else f"{prefix}.{it.name}"
                result[full] = module_hash(it)
                _walk(it.items, full)
            elif isinstance(it, A.ModuleDecl):
                full = ".".join(it.path)
                if prefix:
                    full = f"{prefix}.{full}"
                result[full] = module_hash(it)

    _walk(prog.items)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _list_structs(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate all top-level
    StructDecls in a source file with their structural-hash + field
    count.

    Output format per struct (one line):
      `<name> fields=N hash=<12hex>`
    Sorted alphabetically by struct name.

    Companion to --list-fns (which does the same for fns). Use for
    repository struct inventory + drift detection.

    Exit 0 always.
    """
    from .ast_hash import structural_hash, short_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    structs = sorted(
        (it for it in prog.items if isinstance(it, A.StructDecl)),
        key=lambda s: s.name,
    )
    for s in structs:
        h = short_hash(structural_hash(s))
        print(f"{s.name} fields={len(s.fields)} hash={h}")
    return 0


def _list_fns_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-fns in machine-
    readable JSON form.

    Output schema:
      {
        "<fn_name>": {"sig_hash": "<64hex>", "body_hash": "<64hex>"},
        ...
      }

    Unlike the human format (--list-fns), this uses FULL 64-hex hashes
    (not short_hash) — JSON consumers typically want full hashes for
    storage / comparison; short hashes are a human-display convenience.
    """
    import json
    from .ast_hash import structural_hash, fn_signature_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    fns = sorted(
        (it for it in prog.items if isinstance(it, A.FnDecl)),
        key=lambda f: f.name,
    )
    result = {
        fn.name: {
            "sig_hash": fn_signature_hash(fn),
            "body_hash": structural_hash(fn),
        }
        for fn in fns
    }
    print(json.dumps(result, sort_keys=True, indent=2))
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

    if "--ast-stats" in flags:
        if len(args) < 1:
            print("usage: --ast-stats <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_ast_stats(args[0]))

    if "--ast-stats-json" in flags:
        if len(args) < 1:
            print("usage: --ast-stats-json <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_ast_stats_json(args[0]))

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

    if "--trace-dump-summary" in flags:
        if len(args) < 1:
            print("usage: --trace-dump-summary <file.json>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_trace_dump_summary(args[0]))

    if "--trace-dump-summary-json" in flags:
        if len(args) < 1:
            print("usage: --trace-dump-summary-json <file.json>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_trace_dump_summary_json(args[0]))

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

    if "--list-fns-json" in flags:
        if len(args) < 1:
            print("usage: --list-fns-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_fns_json(args[0]))

    if "--list-structs" in flags:
        if len(args) < 1:
            print("usage: --list-structs <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_structs(args[0]))

    if "--list-structs-json" in flags:
        if len(args) < 1:
            print("usage: --list-structs-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_structs_json(args[0]))

    if "--list-modules-json" in flags:
        if len(args) < 1:
            print("usage: --list-modules-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_modules_json(args[0]))

    if "--parse-only" in flags:
        if len(args) < 1:
            print("usage: --parse-only <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_parse_only(args[0]))

    if "--list-fn-attrs" in flags:
        if len(args) < 1:
            print("usage: --list-fn-attrs <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_fn_attrs(args[0]))

    if "--fn-callgraph" in flags:
        if len(args) < 2:
            print("usage: --fn-callgraph <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph(args[0], args[1]))

    if "--fn-callers" in flags:
        if len(args) < 2:
            print("usage: --fn-callers <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callers(args[0], args[1]))

    if "--fn-callgraph-all" in flags:
        if len(args) < 1:
            print("usage: --fn-callgraph-all <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph_all(args[0]))

    if "--fn-callers-all" in flags:
        if len(args) < 1:
            print("usage: --fn-callers-all <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callers_all(args[0]))

    if "--fn-reachable-from" in flags:
        if len(args) < 2:
            print("usage: --fn-reachable-from <file.hx> <entry_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_reachable_from(args[0], args[1]))

    if "--fn-reachable-to" in flags:
        if len(args) < 2:
            print("usage: --fn-reachable-to <file.hx> <target_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_reachable_to(args[0], args[1]))

    if "--fn-call-stats" in flags:
        if len(args) < 1:
            print("usage: --fn-call-stats <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_call_stats(args[0]))

    if "--fn-leaves" in flags:
        if len(args) < 1:
            print("usage: --fn-leaves <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_leaves(args[0]))

    if "--fn-roots" in flags:
        if len(args) < 1:
            print("usage: --fn-roots <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_roots(args[0]))

    if "--fn-recursive" in flags:
        if len(args) < 1:
            print("usage: --fn-recursive <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_recursive(args[0]))

    if "--fn-cycles" in flags:
        if len(args) < 1:
            print("usage: --fn-cycles <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_cycles(args[0]))

    if "--list-fn-attrs-json" in flags:
        if len(args) < 1:
            print("usage: --list-fn-attrs-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_fn_attrs_json(args[0]))

    if "--list-fns-by-attr" in flags:
        if len(args) < 2:
            print("usage: --list-fns-by-attr <file.hx> <attr>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_fns_by_attr(args[0], args[1]))

    if "--check-program-hash" in flags:
        if len(args) < 2:
            print("usage: --check-program-hash <file.hx> "
                  "<expected_hex_hash>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_check_program_hash(args[0], args[1]))

    if "--check-program-hash-from-file" in flags:
        if len(args) < 2:
            print("usage: --check-program-hash-from-file <file.hx> "
                  "<expected_hash_file>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_check_program_hash_from_file(args[0], args[1]))

    if "--check-program-signature-hash" in flags:
        if len(args) < 2:
            print("usage: --check-program-signature-hash <file.hx> "
                  "<expected_hex_hash>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_check_program_signature_hash(args[0], args[1]))

    if "--check-program-signature-hash-from-file" in flags:
        if len(args) < 2:
            print("usage: --check-program-signature-hash-from-file "
                  "<file.hx> <expected_hash_file>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_check_program_signature_hash_from_file(
            args[0], args[1]))

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

    if "--autotune-summary-json" in flags:
        if len(args) < 1:
            print("usage: --autotune-summary-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_autotune_summary_json(args[0]))

    if "--validate-autotune" in flags:
        if len(args) < 1:
            print("usage: --validate-autotune <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_autotune(args[0]))

    if "--validate-autotune-json" in flags:
        if len(args) < 1:
            print("usage: --validate-autotune-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_autotune_json(args[0]))

    if "--validate-trace-attrs" in flags:
        if len(args) < 1:
            print("usage: --validate-trace-attrs <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_trace_attrs(args[0]))

    if "--list-traced-fns" in flags:
        if len(args) < 1:
            print("usage: --list-traced-fns <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_traced_fns(args[0]))

    if "--list-traced-fns-json" in flags:
        if len(args) < 1:
            print("usage: --list-traced-fns-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_traced_fns_json(args[0]))

    if "--validate-all" in flags:
        if len(args) < 1:
            print("usage: --validate-all <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_all(args[0]))

    if "--validate-all-json" in flags:
        if len(args) < 1:
            print("usage: --validate-all-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_all_json(args[0]))

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

    if "--list-pytrees-json" in flags:
        if len(args) < 1:
            print("usage: --list-pytrees-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_pytrees_json(args[0]))

    if "--pytree-leaf-paths" in flags:
        if len(args) < 2:
            print("usage: --pytree-leaf-paths <file.hx> <struct_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_pytree_leaf_paths(args[0], args[1]))

    if "--validate-pytrees" in flags:
        if len(args) < 1:
            print("usage: --validate-pytrees <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_pytrees(args[0]))

    if "--validate-pytrees-json" in flags:
        if len(args) < 1:
            print("usage: --validate-pytrees-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_pytrees_json(args[0]))

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
