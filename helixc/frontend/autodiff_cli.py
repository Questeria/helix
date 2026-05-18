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
    --dump-ast-hashes-json <file.hx>
        Same as --dump-ast-hashes but JSON {fn_name: 64hex} dict.
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
    --diff-trace-json <a.json> <b.json>
        Same as --diff-trace but machine-readable JSON output.
    --trace-dump-summary <file.json>
        High-level stats of a trace dump: counts, balance, short hash.
    --trace-dump-summary-json <file.json>
        Same as --trace-dump-summary but machine-readable JSON output.
    --diff-program-hash <a.hx> <b.hx>
        Compare two programs: prints SAME or DIFFER + per-fn breakdown.
    --diff-program-hash-json <a.hx> <b.hx>
        Same as --diff-program-hash but JSON with match flag, both
        program + signature hashes, and kind discriminator.
    --changed-fns <a.hx> <b.hx>
        List fns whose body hash differs between two files.
    --changed-fns-json <a.hx> <b.hx>
        Same as --changed-fns but JSON with added/removed/modified
        and full 64-hex hashes.
    --fn-sig-hash <file.hx> <fn_name>
        Print the signature-only hash (ABI-affecting fields only).
    --fn-signature <file.hx> <fn_name>
        Print the source-level signature of a fn (one line).
    --fn-signature-json <file.hx> <fn_name>
        Same as --fn-signature but full structured JSON (generics,
        params, return_ty, attrs, is_pub/is_extern/extern_abi).
    --list-fns <file.hx>
        Enumerate all fns with sig + body hash columns.
    --list-fns-json <file.hx>
        Same as --list-fns but machine-readable JSON (full 64-hex hashes).
    --list-structs <file.hx>
        Enumerate all structs with field count + content hash.
    --list-uses <file.hx>
        Enumerate `use` decls (imports) as dot-joined paths.
    --list-uses-json <file.hx>
        Same as --list-uses but JSON output with raw segments.
    --list-consts <file.hx>
        Enumerate top-level ConstDecls as '<name>: <ty>' lines.
    --list-enums <file.hx>
        Enumerate top-level EnumDecls as '<name> variants=N' lines.
    --list-enums-json <file.hx>
        Same as --list-enums but JSON with variant names included.
    --list-type-aliases <file.hx>
        Enumerate top-level TypeAlias decls as '<name> = <ty>' lines.
    --list-type-aliases-json <file.hx>
        Same as --list-type-aliases but machine-readable JSON output.
    --list-agents <file.hx>
        Enumerate top-level AgentDecls as '<name> methods=N' lines.
    --list-agents-json <file.hx>
        Same as --list-agents but JSON with method names included.
    --list-impls <file.hx>
        Enumerate top-level ImplBlocks as '<target> methods=N' (or
        '<trait> for <target> methods=N' for trait impls) lines.
    --list-impls-json <file.hx>
        Same as --list-impls but JSON with method names included.
    --impl-methods <file.hx> <target>
        Print method signatures of every impl block for <target>
        (params + return ty). Merges multiple impl blocks.
    --impl-methods-json <file.hx> <target>
        Same as --impl-methods but JSON with trait field included.
    --agent-methods <file.hx> <agent_name>
        Print method signatures of an agent (params + return ty).
    --agent-methods-json <file.hx> <agent_name>
        Same as --agent-methods but machine-readable JSON output.
    --type-alias-target <file.hx> <alias_name>
        Print the target type of a TypeAlias.
    --type-alias-target-json <file.hx> <alias_name>
        Same as --type-alias-target but JSON {name, target} output.
    --enum-variants <file.hx> <enum_name>
        Print per-variant lines for an enum (with payload types if any).
    --enum-variants-json <file.hx> <enum_name>
        Same as --enum-variants but JSON with payload_tys list.
    --list-consts-json <file.hx>
        Same as --list-consts but machine-readable JSON output.
    --const-value <file.hx> <const_name>
        Print the literal value of a top-level ConstDecl.
    --const-value-json <file.hx> <const_name>
        Same as --const-value but JSON {name, ty, value} (typed).
    --struct-fields <file.hx> <struct_name>
        Print '<name>: <ty>' per field (declaration order, not sorted).
    --struct-fields-json <file.hx> <struct_name>
        Same as --struct-fields but machine-readable JSON output.
    --list-structs-json <file.hx>
        Same as --list-structs but machine-readable JSON output.
    --list-modules-json <file.hx>
        Same as --list-modules but machine-readable JSON output.
    --module-stats <file.hx> <mod_name>
        Per-ModBlock item-count introspection (fns/structs/enums/
        etc. nested inside a named module). Dotted names supported.
    --module-stats-json <file.hx> <mod_name>
        Same as --module-stats but JSON with all 9 keys (zeros incl.).
    --module-hash-json <file.hx> <mod_name>
        Same as --module-hash but JSON {name, hash} output.
    --validate-trace-attrs-json <file.hx>
        Same as --validate-trace-attrs but JSON {ok, violations}.
    --parse-only <file.hx>
        Lightest CI gate: exit 0 if source parses cleanly, 1 on error.
    --list-fn-attrs <file.hx>
        Enumerate all fns with their attribute list (@pure, @trace, etc).
    --fn-callgraph <file.hx> <fn_name>
        Print fn names directly called from inside <fn_name>'s body.
    --fn-callgraph-json <file.hx> <fn_name>
        Same as --fn-callgraph but JSON {caller, callees: [...]}.
    --fn-callers-json <file.hx> <fn_name>
        Same as --fn-callers but JSON {target, callers: [...]}.
    --fn-reachable-from-json <file.hx> <entry_fn>
        Same as --fn-reachable-from but JSON {entry, reachable, n}.
    --fn-reachable-to-json <file.hx> <target_fn>
        Same as --fn-reachable-to but JSON {target, reachable, n}.
    --fn-body-stats <file.hx> <fn_name>
        Per-fn body AST-node counts (calls/binops/ifs/loops/matches).
    --fn-body-stats-json <file.hx> <fn_name>
        Same as --fn-body-stats but machine-readable JSON output.
    --fn-body-stats-all <file.hx>
        Per-fn body-stats for every fn in the file as JSON profile.
    --fn-body-stats-summary <file.hx>
        Aggregate body-stats summary (total/max/min/avg per metric).
    --fn-body-stats-rank <file.hx> <metric> <top_n>
        Top-N fns by a metric (hotspot ranking).
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
    --fn-callgraph-depth <file.hx> <entry_fn>
        Max acyclic stack depth from entry_fn (cycles clipped).
    --fn-callgraph-depth-all <file.hx>
        Whole-program {fn: depth} JSON profile for stack-risk ranking.
    --fn-topo-sort <file.hx>
        Topological sort: leaves first, then their dependents.
    --fn-isolated <file.hx>
        List orphan fns: no callers AND no callees (strongest dead-code).
    --fn-call-path <file.hx> <from_fn> <to_fn>
        Shortest call chain from <from_fn> to <to_fn> via BFS.
    --fn-distance <file.hx> <from_fn> <to_fn>
        Edge-count distance from <from_fn> to <to_fn> (-1 if no path).
    --fn-distance-matrix <file.hx>
        Whole-program pairwise shortest-path distances as JSON.
    --fn-callgraph-summary <file.hx>
        High-level structural overview JSON (counts/depth/diameter/SCCs).
    --fn-callgraph-dot <file.hx>
        Emit callgraph as Graphviz .dot (pipe through `dot -Tpng`).
    --fn-callgraph-mermaid <file.hx>
        Emit callgraph as Mermaid flowchart (markdown-embeddable).
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
    --list-fns-by-attr-json <file.hx> <attr>
        Same as --list-fns-by-attr but machine-readable JSON output.
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
    --pytree-shape-json <file.hx> <struct_name>
        Same as --pytree-shape but machine-readable JSON output.
    --pytree-leaf-paths-json <file.hx> <struct_name>
        Same as --pytree-leaf-paths but JSON output.
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


def _dump_ast_hashes_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --dump-ast-hashes in
    machine-readable JSON form.

    Output schema:
      {"<fn_name>": "<64hex>", ...}

    Like --list-fns-json this uses FULL 64-hex hashes (not short_hash).
    Differs from --list-fns-json in that the values are just the body
    hash (no sig_hash), matching the structural-hash-only semantics of
    the text form.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    result = {
        it.name: structural_hash(it)
        for it in prog.items if isinstance(it, A.FnDecl)
    }
    print(json.dumps(result, sort_keys=True, indent=2))
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


def _diff_trace_json(path_a: str, path_b: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: machine-readable diff
    between two trace JSON dumps.

    Output schema:
      {"status": "MATCH" | "DIFFER",
       "events_a": <int>,
       "events_b": <int>,
       "first_divergence": {        // only present if DIFFER
         "index": <int>,
         "a": {... event dict ...} or null,
         "b": {... event dict ...} or null
       }
      }

    Exit codes match --diff-trace: 0 for MATCH or for any DIFFER
    output (so JSON-consuming tooling parses every case uniformly);
    1 reserved for I/O errors only.
    """
    import json
    from .trace_pass import (
        trace_from_canonical_json, trace_equiv, trace_diff,
    )
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
    result: dict = {
        "events_a": len(a),
        "events_b": len(b),
    }
    if trace_equiv(a, b):
        result["status"] = "MATCH"
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    result["status"] = "DIFFER"
    diff = trace_diff(a, b)
    if diff is not None:
        idx, ea, eb = diff

        def _ev(ev) -> dict:
            if ev is None:
                return None
            return {
                "op_kind": ev.op_kind,
                "fn_name": ev.fn_name,
                "operands": list(ev.operands),
                "result": ev.result,
            }

        result["first_divergence"] = {
            "index": idx,
            "a": _ev(ea),
            "b": _ev(eb),
        }
    print(json.dumps(result, sort_keys=True, indent=2))
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


def _pytree_shape_json(path: str, struct_name: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: --pytree-shape in
    machine-readable JSON form.

    Output schema:
      {
        "leaves": [
          {"path": "<dotted>", "ty": "<prim>", "diff": <bool>}, ...
        ],
        "total": <int>,
        "diff": <int>,
        "non_diff": <int>
      }
    Leaves sorted by path.

    Exit 0 success; 1 if struct not found or pytree-rejected.
    """
    import json
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
    leaves_sorted = sorted(leaves, key=lambda l: l.path)
    diff = sum(1 for l in leaves if l.is_diff)
    result = {
        "leaves": [
            {"path": l.path, "ty": l.ty_name, "diff": l.is_diff}
            for l in leaves_sorted
        ],
        "total": len(leaves),
        "diff": diff,
        "non_diff": len(leaves) - diff,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _pytree_leaf_paths_json(path: str, struct_name: str) -> int:
    """Stage 59 follow-on / Tier 2 #7 polish: --pytree-leaf-paths in
    machine-readable JSON form.

    Output schema:
      {"paths": ["<path1>", "<path2>", ...]}
    Paths sorted alphabetically.
    """
    import json
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
    paths = sorted(l.path for l in leaves)
    print(json.dumps({"paths": paths}, sort_keys=True, indent=2))
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


def _validate_trace_attrs_json(path: str) -> int:
    """Stage 59 follow-on / Tier 3 #11 polish: --validate-trace-attrs
    in machine-readable JSON form. Completes JSON-parity for all
    4 per-system validator gates (pytrees, autotune, trace-attrs,
    all-aggregator).

    Output schema:
      {"ok": bool, "violations": ["<diag1>", ...]}
    Exits 0 on clean (ok=true), 1 on rule violation (ok=false).
    """
    import json
    from .trace_pass import validate_trace_attrs
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    diags = validate_trace_attrs(prog)
    result = {"ok": not diags, "violations": [str(d) for d in diags]}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if not diags else 1


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


def _module_hash_json_cli(path: str, mod_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --module-hash in
    machine-readable JSON form.

    Output schema:
      {"name": "<mod_name>", "hash": "<64hex>"}
    """
    import json
    from .ast_hash import module_hash
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _find(items, target: str, prefix: str = ""):
        for it in items:
            if isinstance(it, A.ModBlock):
                full = it.name if not prefix else f"{prefix}.{it.name}"
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
    print(json.dumps({"name": mod_name, "hash": module_hash(mod)},
                      sort_keys=True, indent=2))
    return 0


def _module_stats(path: str, mod_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-ModBlock item-count
    introspection. Print counts of nested fns / structs / enums /
    type-aliases / uses / consts / agents / impls / sub-modules within
    a named module.

    Output: one line per non-zero category as '<kind>: N' (alphabetic
    order). Empty module → no output.

    Per-item introspection family extends to 8 Item subclasses:
    struct-fields, const-value, enum-variants, agent-methods,
    type-alias-target, impl-methods, fn-signature, module-stats.

    For nested modules, accept dotted names (e.g., 'outer.inner').

    Exit 0 success, 1 if mod_name not found.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _find(items, target: str, prefix: str = ""):
        for it in items:
            if isinstance(it, A.ModBlock):
                full = it.name if not prefix else f"{prefix}.{it.name}"
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
        print(f"error: autodiff_cli: module {mod_name!r} not found "
              f"in {path}", file=sys.stderr)
        return 1
    # ModuleDecl is a header-only declaration; no items to count.
    items = getattr(mod, "items", None) or []
    counts: dict[str, int] = {
        "fns": sum(1 for it in items if isinstance(it, A.FnDecl)),
        "structs": sum(1 for it in items if isinstance(it, A.StructDecl)),
        "enums": sum(1 for it in items if isinstance(it, A.EnumDecl)),
        "type_aliases": sum(1 for it in items if isinstance(it, A.TypeAlias)),
        "uses": sum(1 for it in items if isinstance(it, A.UseDecl)),
        "consts": sum(1 for it in items if isinstance(it, A.ConstDecl)),
        "agents": sum(1 for it in items if isinstance(it, A.AgentDecl)),
        "impls": sum(1 for it in items if isinstance(it, A.ImplBlock)),
        "modules": sum(1 for it in items if isinstance(it, A.ModBlock)),
    }
    for kind in sorted(counts):
        if counts[kind] > 0:
            print(f"{kind}: {counts[kind]}")
    return 0


def _module_stats_json(path: str, mod_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --module-stats in
    machine-readable JSON form.

    Output schema:
      {"name": "<mod_name>",
       "fns": N, "structs": N, "enums": N, "type_aliases": N,
       "uses": N, "consts": N, "agents": N, "impls": N,
       "modules": N}
    All 9 keys always present (zeros included).
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _find(items, target: str, prefix: str = ""):
        for it in items:
            if isinstance(it, A.ModBlock):
                full = it.name if not prefix else f"{prefix}.{it.name}"
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
        print(f"error: autodiff_cli: module {mod_name!r} not found "
              f"in {path}", file=sys.stderr)
        return 1
    items = getattr(mod, "items", None) or []
    result = {
        "name": mod_name,
        "fns": sum(1 for it in items if isinstance(it, A.FnDecl)),
        "structs": sum(1 for it in items if isinstance(it, A.StructDecl)),
        "enums": sum(1 for it in items if isinstance(it, A.EnumDecl)),
        "type_aliases": sum(1 for it in items if isinstance(it, A.TypeAlias)),
        "uses": sum(1 for it in items if isinstance(it, A.UseDecl)),
        "consts": sum(1 for it in items if isinstance(it, A.ConstDecl)),
        "agents": sum(1 for it in items if isinstance(it, A.AgentDecl)),
        "impls": sum(1 for it in items if isinstance(it, A.ImplBlock)),
        "modules": sum(1 for it in items if isinstance(it, A.ModBlock)),
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


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


def _list_fns_by_attr_json(path: str, attr: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-fns-by-attr in
    machine-readable JSON form.

    Output schema:
      {"attr": "<attr>", "fns": ["fn1", "fn2", ...]}
    Sorted alphabetically.
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    matching = sorted(
        f.name for f in iter_fn_decls(prog)
        if attr in f.attrs
    )
    print(json.dumps({"attr": attr, "fns": matching},
                      sort_keys=True, indent=2))
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


def _fn_callgraph_mermaid(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: emit the local callgraph
    as Mermaid flowchart syntax. Drop-in for markdown docs (GitHub
    auto-renders Mermaid in .md files).

    Output:
      ```
      flowchart LR
          fn_a --> fn_b
          fn_a --> fn_c
          ...
      ```
    Edges sorted alphabetically for deterministic output.

    Pair with --fn-callgraph-dot (Graphviz). Together they cover
    the two dominant text-based graph-viz formats. Use case:
    embed in README to auto-display the callgraph on the project
    homepage.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = sorted(fn.name for fn in all_fns)
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
        graph[fn.name] = {c for c in callees if c in set(all_names)}

    print("flowchart LR")
    for src_name in all_names:
        for dst in sorted(graph.get(src_name, set())):
            print(f"    {src_name} --> {dst}")
    return 0


def _fn_callgraph_dot(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: emit the local callgraph
    as a Graphviz .dot format string.

    Output: standard digraph syntax:
      digraph callgraph {
          "fn_a" -> "fn_b";
          "fn_a" -> "fn_c";
          ...
      }
    Edges sorted alphabetically for deterministic output (so diffs
    across versions are stable).

    Use case: pipe through `dot -Tpng > callgraph.png` for visual
    inspection. Pairs with --fn-callgraph-all (JSON) and
    --fn-callgraph-summary (numeric overview) — three formats for
    three audiences: tooling (JSON), humans (Graphviz), dashboards
    (summary).

    Pivots the call-graph sub-arc to a new 'output format' axis
    (text/JSON were already covered; this is visualization).
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = sorted(fn.name for fn in all_fns)
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
        graph[fn.name] = {c for c in callees if c in set(all_names)}

    print("digraph callgraph {")
    for src_name in all_names:
        for dst in sorted(graph.get(src_name, set())):
            print(f'    "{src_name}" -> "{dst}";')
    print("}")
    return 0


def _fn_callgraph_summary(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: high-level structural
    overview of the callgraph as JSON. One read gives the whole shape.

    Output schema:
      {
        "fns": <int>,           # total fn count
        "edges": <int>,          # total directed call edges
        "leaves": <int>,         # fns with no callees
        "roots": <int>,          # fns with no callers
        "isolated": <int>,       # fns with neither (leaves ∩ roots)
        "recursive": <int>,      # fns with direct self-calls
        "sccs_nontrivial": <int>, # mutual-recursion cycles (size>=2)
        "max_depth": <int>,      # longest acyclic chain (any entry)
        "diameter": <int>,       # max pairwise distance (0 if disconnected)
      }

    Pure summary derivation from primitives already shipped. Companion
    to --ast-stats but specifically for the callgraph structure.

    Use case: code-health dashboard input — one number per metric per
    file, easy to track over time.
    """
    import json
    from collections import deque
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}

    # Build adjacency (including self-loops for recursive count).
    graph_self: dict[str, set] = {}
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
        graph_self[fn.name] = {c for c in callees if c in all_names}

    # graph_no_self for non-recursive analysis (cycles / topology).
    graph: dict[str, set] = {n: (s - {n}) for n, s in graph_self.items()}

    # Counts.
    edges = sum(len(s) for s in graph_self.values())
    leaves = sum(1 for n in all_names if not graph_self[n])
    in_count: dict[str, int] = {n: 0 for n in all_names}
    for n, callees in graph_self.items():
        for c in callees:
            if c != n:  # don't count self-edge as in-edge
                in_count[c] += 1
    roots = sum(1 for n in all_names if in_count[n] == 0)
    isolated = sum(1 for n in all_names
                   if in_count[n] == 0 and not graph_self[n])
    recursive = sum(1 for n in all_names if n in graph_self[n])

    # Tarjan SCC for mutual cycles.
    sccs_nontrivial = 0
    index_counter = [0]
    stack: list = []
    on_stack: set = set()
    index_d: dict = {}
    lowlink: dict = {}

    def _scc(v: str) -> None:
        nonlocal sccs_nontrivial
        index_d[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, set()):
            if w not in index_d:
                _scc(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index_d[w])
        if lowlink[v] == index_d[v]:
            comp: list = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) >= 2:
                sccs_nontrivial += 1

    for v in sorted(graph.keys()):
        if v not in index_d:
            _scc(v)

    # Max depth (longest acyclic chain over any entry).
    def _depth(start: str) -> int:
        seen: set = set()

        def _go(v: str) -> int:
            if v in seen:
                return 0
            seen.add(v)
            try:
                children = graph.get(v, set()) - seen
                if not children:
                    return 1
                return 1 + max((_go(c) for c in children), default=0)
            finally:
                seen.discard(v)

        return _go(start)

    max_depth = max((_depth(n) for n in all_names), default=0)

    # Diameter via BFS from each node.
    def _bfs_max(start: str) -> int:
        dist: dict = {start: 0}
        queue = deque([start])
        m = 0
        while queue:
            v = queue.popleft()
            for w in graph.get(v, set()):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                    if dist[w] > m:
                        m = dist[w]
        return m

    diameter = max((_bfs_max(n) for n in all_names), default=0)

    summary = {
        "fns": len(all_names),
        "edges": edges,
        "leaves": leaves,
        "roots": roots,
        "isolated": isolated,
        "recursive": recursive,
        "sccs_nontrivial": sccs_nontrivial,
        "max_depth": max_depth,
        "diameter": diameter,
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def _fn_distance_matrix(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: whole-program pairwise
    shortest-path distance matrix as JSON.

    Output schema:
      {
        "<from_fn>": {"<to_fn>": <distance_int>, ...},
        ...
      }
    where <distance_int> is the edge-count shortest distance (0 for
    self, -1 for unreachable). All locally-defined fns appear as
    both outer and inner keys.

    Use cases:
    - Cluster fns by mutual distance (community detection)
    - Compute eccentricity (max distance from a fn to anywhere
      reachable) per fn for tier classification
    - Pair with --fn-distance for single-pair queries; this is the
      whole-program JSON profile for batch consumption.

    Complexity: O(V * (V + E)) — N BFSes. Fine for source files up
    to a few thousand fns.
    """
    import json
    from collections import deque
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = sorted(fn.name for fn in all_fns)
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
        graph[fn.name] = {c for c in callees if c in set(all_names)}

    def _bfs_distances(start: str) -> dict:
        """Return {target_name: distance} for every reachable target.
        Unreachable targets get -1; self gets 0."""
        dist: dict = {n: -1 for n in all_names}
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for w in graph.get(v, set()):
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        return dist

    matrix = {name: _bfs_distances(name) for name in all_names}
    print(json.dumps(matrix, sort_keys=True, indent=2))
    return 0


def _fn_distance(path: str, from_fn: str, to_fn: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: shortest-path distance
    (edge count) from from_fn to to_fn in the local callgraph. -1
    if no path. 0 if from_fn == to_fn.

    Output: a single integer. Tooling-friendly: pipe into shell
    arithmetic or sort.

    Use case: 'how far is X from main?' — pair with --fn-callgraph-
    depth-all to rank fns by max depth they're called at.

    Exit codes:
      0 — success (distance printed, possibly -1)
      1 — either from_fn or to_fn not found in source
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    for needed in (from_fn, to_fn):
        if needed not in all_names:
            print(f"error: autodiff_cli: fn {needed!r} not found in {path}",
                  file=sys.stderr)
            return 1

    if from_fn == to_fn:
        print(0)
        return 0

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

    # BFS layered distance.
    visited: set = {from_fn}
    layer: list = [from_fn]
    distance = 0
    while layer:
        distance += 1
        next_layer: list = []
        for v in layer:
            for w in graph.get(v, set()):
                if w == to_fn:
                    print(distance)
                    return 0
                if w not in visited:
                    visited.add(w)
                    next_layer.append(w)
        layer = next_layer
    print(-1)
    return 0


def _fn_call_path(path: str, from_fn: str, to_fn: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: shortest call chain
    from `from_fn` to `to_fn` via BFS through the local callgraph.

    Output: the path as ' -> '-separated fn names on one line.
    If no path exists, prints 'no path' and exits 0 (not 1 — absent
    path is a valid query result, not an error).

    Use cases:
    - 'How does fn A end up calling fn B?' — see the call chain
    - Witness path for impact-zone analysis (--fn-reachable-from
      tells you B is reachable; this tells you HOW)
    - Refactor-step planning: shortest intercept point

    Exit codes:
      0 — success (path found OR 'no path' reported)
      1 — either from_fn or to_fn not found in source
      2 — bad args
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    for needed in (from_fn, to_fn):
        if needed not in all_names:
            print(f"error: autodiff_cli: fn {needed!r} not found in {path}",
                  file=sys.stderr)
            return 1

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

    # BFS with predecessor tracking.
    if from_fn == to_fn:
        print(from_fn)
        return 0
    visited: set = {from_fn}
    pred: dict[str, str] = {}
    queue: list = [from_fn]
    found = False
    while queue:
        next_queue: list = []
        for v in queue:
            for w in graph.get(v, set()):
                if w not in visited:
                    visited.add(w)
                    pred[w] = v
                    if w == to_fn:
                        found = True
                        break
                    next_queue.append(w)
            if found:
                break
        if found:
            break
        queue = next_queue

    if not found:
        print("no path")
        return 0

    # Reconstruct the path from to_fn back to from_fn.
    path_nodes: list = [to_fn]
    cur = to_fn
    while cur != from_fn:
        cur = pred[cur]
        path_nodes.append(cur)
    path_nodes.reverse()
    print(" -> ".join(path_nodes))
    return 0


def _fn_isolated(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: list 'isolated' fns —
    fns with NO callers AND NO callees in the local callgraph (truly
    orphan / standalone in this file).

    Equivalent to: leaves ∩ roots. A fn is isolated if it neither
    calls any other fn nor is called by any local fn.

    Use cases:
    - Strongest dead-code candidates (no incoming + no outgoing
      edges → almost certainly safe to remove unless used via FFI)
    - Smallest possible refactor units (no transitive impact)
    - Test-fn discovery (in some conventions, test fns have no
      callers and are picked up via runtime registration)

    Sorted alphabetically, one per line.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    fan_out: dict[str, set] = {fn.name: set() for fn in all_fns}
    fan_in: dict[str, set] = {fn.name: set() for fn in all_fns}

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

    isolated = sorted(
        name for name in all_names
        if not fan_in[name] and not fan_out[name]
    )
    for name in isolated:
        print(name)
    return 0


def _fn_topo_sort(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: topological sort of the
    callgraph. Output fns in leaves-first order — every fn appears
    AFTER all of its callees.

    One fn per line. Cycles are broken by SCC condensation: each SCC
    is emitted as a contiguous block (alphabetically sorted within
    the SCC), and the blocks themselves are topologically ordered.

    Use cases:
    - 'What order should I read this codebase?' — leaves are the
      simplest building blocks; later fns compose earlier ones
    - Inlining order: inline leaves into callers, then propagate
    - Bottom-up proof construction: prove termination/correctness
      starting at leaves, build up to entry points

    Output for typical cases: a permutation of all fn names such that
    if A calls B (directly), then B appears before A.
    """
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
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

    # Tarjan SCC → produces SCCs in reverse topo order (sinks first).
    index_counter = [0]
    stack: list[str] = []
    on_stack: set = set()
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
            sccs.append(scc)

    for v in sorted(graph.keys()):
        if v not in index:
            _strongconnect(v)

    # sccs comes out in reverse-topo (sinks first). That's what we
    # want: leaves first. Within each SCC, sort alphabetically.
    for scc in sccs:
        for name in sorted(scc):
            print(name)
    return 0


def _fn_callgraph_depth_all(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: whole-program stack-
    depth profile as JSON. For every fn, compute the max acyclic
    call-stack depth treating that fn as entry.

    Output schema:
      {"<fn_name>": <depth>, ...}

    Companion to --fn-callgraph-depth (single fn). Use cases:
    - Rank all fns by depth to find the deepest call chains in
      the program — stack-overflow risk inventory
    - Sort fns by depth as a complexity proxy (deep = more callees
      transitively, harder to reason about)
    - Track depth distribution across releases as a code-health
      metric
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}

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

    def _depth(start: str) -> int:
        on_stack: set = set()

        def _go(v: str) -> int:
            if v in on_stack:
                return 0
            on_stack.add(v)
            try:
                children = graph.get(v, set()) - on_stack
                if not children:
                    return 1
                return 1 + max((_go(c) for c in children), default=0)
            finally:
                on_stack.discard(v)

        return _go(start)

    result = {name: _depth(name) for name in all_names}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _fn_callgraph_depth(path: str, entry_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: max acyclic call-stack
    depth from a fn entry point.

    Computes the longest call chain (in terms of distinct fn-frames)
    reachable from `entry_name`, treating the callgraph as a DAG by
    breaking cycles at the first re-entry. The entry counts as depth
    1; a fn that calls one other fn with no further calls gives
    depth 2; etc.

    Phase-0 limitation: cycles are clipped (the SCC is treated as
    a single node for the longest-path computation). Use --fn-cycles
    to identify the cycles.

    Output: a single integer (the depth). Exit 0 success, 1 if
    entry_name not found.

    Use case: stack-overflow-risk audit. A fn with depth = 50 might
    risk overflow on small embedded stacks; pair with TCO analysis.
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

    # Build forward adjacency (local fns only).
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

    # DFS-with-on-stack-set for longest acyclic path.
    on_stack: set = set()

    def _depth(v: str) -> int:
        if v in on_stack:
            return 0  # cycle: clip
        on_stack.add(v)
        try:
            children = graph.get(v, set()) - on_stack
            if not children:
                return 1
            return 1 + max((_depth(c) for c in children), default=0)
        finally:
            on_stack.discard(v)

    print(_depth(entry_name))
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


def _fn_reachable_to_json(path: str, target_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --fn-reachable-to in
    machine-readable JSON form.

    Output schema:
      {"target": "<fn_name>",
       "reachable": ["<f1>", "<f2>", ...],
       "n_reachable": N}
    Includes target_name in the reachable list (matches text form).

    Exit 0 on success, 1 if target_name not found.
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    if target_name not in all_names:
        print(f"error: autodiff_cli: fn {target_name!r} not found in {path}",
              file=sys.stderr)
        return 1

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
    sorted_reachable = sorted(reachable)
    print(json.dumps(
        {"target": target_name, "reachable": sorted_reachable,
         "n_reachable": len(sorted_reachable)},
        sort_keys=True, indent=2))
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


def _fn_reachable_from_json(path: str, entry_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --fn-reachable-from in
    machine-readable JSON form.

    Output schema:
      {"entry": "<fn_name>",
       "reachable": ["<f1>", "<f2>", ...],
       "n_reachable": N}
    Includes entry_name in the reachable list (matches text form).

    Exit 0 on success, 1 if entry_name not found.
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    all_fns = list(iter_fn_decls(prog))
    all_names = {fn.name for fn in all_fns}
    if entry_name not in all_names:
        print(f"error: autodiff_cli: fn {entry_name!r} not found in {path}",
              file=sys.stderr)
        return 1

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
    sorted_reachable = sorted(reachable)
    print(json.dumps(
        {"entry": entry_name, "reachable": sorted_reachable,
         "n_reachable": len(sorted_reachable)},
        sort_keys=True, indent=2))
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


def _fn_callers_json(path: str, target_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --fn-callers in
    machine-readable JSON form.

    Output schema:
      {"target": "<fn_name>", "callers": ["<f1>", "<f2>", ...]}
    Callers alphabetically sorted (matches text form).

    Exit 0 always (matches text-form semantics: target_name need
    not exist in the file; empty callers is valid output).
    """
    import json
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
                if (isinstance(n.callee, A.Name)
                        and n.callee.name == target_name):
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
    result = {"target": target_name, "callers": sorted(callers)}
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


def _fn_body_stats_rank(path: str, metric: str, top_n_str: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: rank top-N fns by a
    specific body-stat metric.

    Args:
        metric: one of ast_nodes, calls, binops, ifs, loops, matches
        top_n: integer count (e.g., '5' for top 5)

    Output: one line per fn, sorted descending by metric value:
        '<fn_name> <metric>=<value>'
    Ties broken alphabetically by fn name. If top_n > #fns, output
    is just #fns lines.

    Use case: 'show me the 5 fns with the most calls' — quick
    hotspot ranking without parsing --fn-body-stats-all JSON.

    Exit 0 on success, 2 on bad metric name or non-int top_n.
    """
    metrics = {"ast_nodes", "calls", "binops", "ifs", "loops", "matches"}
    if metric not in metrics:
        print(f"error: autodiff_cli: unknown metric {metric!r} "
              f"(valid: {sorted(metrics)})", file=sys.stderr)
        return 2
    try:
        top_n = int(top_n_str)
    except ValueError:
        print(f"error: autodiff_cli: top_n {top_n_str!r} not an int",
              file=sys.stderr)
        return 2

    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    per_fn: list[tuple[str, int]] = []
    for fn in iter_fn_decls(prog):
        count = 0

        def _walk(node) -> None:
            nonlocal count
            if node is None:
                return
            if hasattr(node, "__dataclass_fields__"):
                if metric == "ast_nodes":
                    count += 1
                elif metric == "calls" and isinstance(node, A.Call):
                    count += 1
                elif metric == "binops" and isinstance(node, A.Binary):
                    count += 1
                elif metric == "ifs" and isinstance(node, A.If):
                    count += 1
                elif metric == "loops" and isinstance(node, (A.For, A.While, A.Loop)):
                    count += 1
                elif metric == "matches" and isinstance(node, A.Match):
                    count += 1
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

        if fn.body is not None:
            _walk(fn.body)
        per_fn.append((fn.name, count))

    # Sort descending by metric value; ties broken alphabetically.
    per_fn.sort(key=lambda x: (-x[1], x[0]))
    for name, value in per_fn[:top_n]:
        print(f"{name} {metric}={value}")
    return 0


def _fn_body_stats_summary(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: aggregate body-stats
    summary across all fns. For each metric, computes total/max/min/
    avg.

    Output schema:
      {
        "<metric>": {"total": N, "max": N, "min": N, "avg": float},
        ...
      }
    Metrics: ast_nodes, calls, binops, ifs, loops, matches.
    avg rounded to 2 decimals; empty file gives all zeros.

    Use case: one-read code-health distribution snapshot for tracking
    complexity drift over time.
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    metrics = ["ast_nodes", "calls", "binops", "ifs", "loops", "matches"]
    per_fn: list[dict] = []

    def _stats(fn) -> dict:
        counts = {k: 0 for k in metrics}

        def _walk(node) -> None:
            if node is None:
                return
            if hasattr(node, "__dataclass_fields__"):
                counts["ast_nodes"] += 1
                if isinstance(node, A.Call):
                    counts["calls"] += 1
                elif isinstance(node, A.Binary):
                    counts["binops"] += 1
                elif isinstance(node, A.If):
                    counts["ifs"] += 1
                elif isinstance(node, (A.For, A.While, A.Loop)):
                    counts["loops"] += 1
                elif isinstance(node, A.Match):
                    counts["matches"] += 1
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

        if fn.body is not None:
            _walk(fn.body)
        return counts

    for fn in iter_fn_decls(prog):
        per_fn.append(_stats(fn))

    n = len(per_fn)
    summary: dict = {}
    for m in metrics:
        values = [f[m] for f in per_fn] or [0]
        total = sum(values)
        summary[m] = {
            "total": total,
            "max": max(values),
            "min": min(values),
            "avg": round(total / n, 2) if n > 0 else 0.0,
        }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def _fn_body_stats_all(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-fn body-stats for
    every fn in the file as JSON.

    Output schema:
      {
        "<fn_name>": {ast_nodes: N, calls: N, binops: N, ifs: N,
                       loops: N, matches: N},
        ...
      }

    Whole-program companion to --fn-body-stats / --fn-body-stats-json.
    Use case: rank fns by complexity in one read; pair with
    --fn-callgraph-depth-all for combined complexity scoring.
    """
    import json
    from .ast_walker import iter_fn_decls
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    result: dict = {}

    def _stats(fn) -> dict:
        counts = {
            "ast_nodes": 0, "calls": 0, "binops": 0,
            "ifs": 0, "loops": 0, "matches": 0,
        }

        def _walk(node) -> None:
            if node is None:
                return
            if hasattr(node, "__dataclass_fields__"):
                counts["ast_nodes"] += 1
                if isinstance(node, A.Call):
                    counts["calls"] += 1
                elif isinstance(node, A.Binary):
                    counts["binops"] += 1
                elif isinstance(node, A.If):
                    counts["ifs"] += 1
                elif isinstance(node, (A.For, A.While, A.Loop)):
                    counts["loops"] += 1
                elif isinstance(node, A.Match):
                    counts["matches"] += 1
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

        if fn.body is not None:
            _walk(fn.body)
        return counts

    for fn in iter_fn_decls(prog):
        result[fn.name] = _stats(fn)

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _fn_body_stats_json(path: str, fn_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --fn-body-stats in
    machine-readable JSON form.

    Output schema:
      {"ast_nodes": N, "calls": N, "binops": N, "ifs": N,
       "loops": N, "matches": N}
    """
    import json
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

    counts = {
        "ast_nodes": 0, "calls": 0, "binops": 0,
        "ifs": 0, "loops": 0, "matches": 0,
    }

    def _walk(node) -> None:
        if node is None:
            return
        if hasattr(node, "__dataclass_fields__"):
            counts["ast_nodes"] += 1
            if isinstance(node, A.Call):
                counts["calls"] += 1
            elif isinstance(node, A.Binary):
                counts["binops"] += 1
            elif isinstance(node, A.If):
                counts["ifs"] += 1
            elif isinstance(node, (A.For, A.While, A.Loop)):
                counts["loops"] += 1
            elif isinstance(node, A.Match):
                counts["matches"] += 1
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
    print(json.dumps(counts, sort_keys=True, indent=2))
    return 0


def _fn_body_stats(path: str, fn_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-fn body-size metrics.
    Counts AST nodes by category as a complexity proxy.

    Output: one stat per line in `key=value` form:
        ast_nodes=N      (total node count via dataclass-walk)
        calls=N          (total A.Call nodes in body)
        binops=N         (total A.Binary nodes)
        ifs=N            (total A.If nodes)
        loops=N          (total A.For + A.While + A.Loop nodes)
        matches=N        (total A.Match nodes)

    Use cases:
    - Identify fns that may need refactoring (high node count)
    - Track per-fn complexity drift across releases
    - Pair with --fn-callgraph-depth for complexity profile
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

    counts = {
        "ast_nodes": 0, "calls": 0, "binops": 0,
        "ifs": 0, "loops": 0, "matches": 0,
    }

    def _walk(node) -> None:
        if node is None:
            return
        if hasattr(node, "__dataclass_fields__"):
            counts["ast_nodes"] += 1
            if isinstance(node, A.Call):
                counts["calls"] += 1
            elif isinstance(node, A.Binary):
                counts["binops"] += 1
            elif isinstance(node, A.If):
                counts["ifs"] += 1
            elif isinstance(node, (A.For, A.While, A.Loop)):
                counts["loops"] += 1
            elif isinstance(node, A.Match):
                counts["matches"] += 1
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

    for k in ("ast_nodes", "calls", "binops", "ifs", "loops", "matches"):
        print(f"{k}={counts[k]}")
    return 0


def _fn_callgraph_json(path: str, fn_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --fn-callgraph in
    machine-readable JSON form.

    Output schema:
      {"caller": "<fn_name>", "callees": ["<f1>", "<f2>", ...]}
    Callees alphabetically sorted (matches text form).

    Exit 0 on success, 1 if fn_name not found.
    """
    import json
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
                if node.callee.segments:
                    callees.add(node.callee.segments[-1])
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
    result = {"caller": fn_name, "callees": sorted(callees)}
    print(json.dumps(result, sort_keys=True, indent=2))
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


def _struct_fields_json(path: str, struct_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --struct-fields in
    machine-readable JSON form.

    Output schema:
      {
        "fields": [{"name": "<name>", "ty": "<ty_string>"}, ...]
      }
    Order matches declaration order. TyGeneric formatted as
    'base<args>' (e.g., 'D<f32>').
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.StructDecl) and it.name == struct_name:
            result = {
                "fields": [
                    {"name": f.name, "ty": _format_ty(f.ty)}
                    for f in it.fields
                ]
            }
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
    print(f"error: autodiff_cli: struct {struct_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _struct_fields(path: str, struct_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print field name + type
    for each field of a struct.

    Output: one field per line, formatted '<name>: <ty>'.
    Order matches struct-declaration order (NOT sorted — declaration
    order matters for ABI / layout).

    Use cases:
    - Quick lookup of struct shape without opening source
    - Pre-flight check for pytree-eligibility audit (verify only
      diff-eligible types)
    - Documentation generation: 'what's in this struct?'

    Exit 0 success, 1 if struct_name not found.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for it in prog.items:
        if isinstance(it, A.StructDecl) and it.name == struct_name:
            for f in it.fields:
                # Resolve type to a human-readable string. TyName has
                # a .name; TyGeneric has a string .base + list .args.
                ty_str = getattr(f.ty, "name", None)
                if ty_str is None:
                    base = getattr(f.ty, "base", None)
                    args = getattr(f.ty, "args", None)
                    # base may itself be either a string (D<f32>) or
                    # a TyName (rare path).
                    base_str = base if isinstance(base, str) else (
                        getattr(base, "name", None) if base else None
                    )
                    if base_str and args:
                        args_strs = []
                        for a in args:
                            an = getattr(a, "name", None)
                            args_strs.append(an if an else repr(a))
                        ty_str = f"{base_str}<{', '.join(args_strs)}>"
                if ty_str is None:
                    ty_str = repr(f.ty)
                print(f"{f.name}: {ty_str}")
            return 0
    print(f"error: autodiff_cli: struct {struct_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _list_uses_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-uses in machine-
    readable JSON form.

    Output schema:
      {"uses": [
        {"path": "<dotted>", "segments": ["<seg1>", "<seg2>", ...]},
        ...
      ]}
    Sorted alphabetically by dotted path. Includes both the joined
    path and the raw segments list for tooling flexibility.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    items: list[dict] = []
    for it in prog.items:
        if isinstance(it, A.UseDecl):
            items.append({
                "path": ".".join(it.path),
                "segments": list(it.path),
            })
    items.sort(key=lambda d: d["path"])
    print(json.dumps({"uses": items}, sort_keys=True, indent=2))
    return 0


def _list_consts_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-consts in
    machine-readable JSON form.

    Output schema:
      {"consts": [{"name": "<name>", "ty": "<ty_string>"}, ...]}
    Declaration order preserved.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    consts = [
        {"name": it.name, "ty": _format_ty(it.ty)}
        for it in prog.items
        if isinstance(it, A.ConstDecl)
    ]
    print(json.dumps({"consts": consts}, sort_keys=True, indent=2))
    return 0


def _const_value_json(path: str, const_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --const-value in
    machine-readable JSON form.

    Output schema:
      {"name": "<name>", "ty": "<ty_string>", "value": <typed_value>}

    Value typing in JSON:
      - IntLit/FloatLit → number
      - BoolLit         → bool
      - StrLit/CharLit  → string
      - Other Expr      → repr() string

    Exit 0 success; 1 if const_name not found.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.ConstDecl) and it.name == const_name:
            v = it.value
            if isinstance(v, A.IntLit):
                json_value = v.value
            elif isinstance(v, A.FloatLit):
                json_value = v.value
            elif isinstance(v, A.BoolLit):
                json_value = v.value
            elif isinstance(v, (A.StrLit, A.CharLit)):
                json_value = v.value
            else:
                json_value = repr(v)
            result = {
                "name": it.name,
                "ty": _format_ty(it.ty),
                "value": json_value,
            }
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
    print(f"error: autodiff_cli: const {const_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _const_value(path: str, const_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print the value
    expression of a specific top-level ConstDecl.

    Output: a single line showing the printable form of the value
    expression. For IntLit/FloatLit/StrLit, prints the literal.
    For more complex expressions, uses `repr(value)` as a fallback.

    Exit 0 success; 1 if const_name not found.

    Use case: quick value lookup. Pair with --list-consts to discover
    names, then this to inspect specific values.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for it in prog.items:
        if isinstance(it, A.ConstDecl) and it.name == const_name:
            v = it.value
            # IntLit / FloatLit / StrLit / BoolLit / CharLit fast path.
            if isinstance(v, A.IntLit):
                print(v.value)
            elif isinstance(v, A.FloatLit):
                print(v.value)
            elif isinstance(v, A.StrLit):
                print(repr(v.value))
            elif isinstance(v, A.BoolLit):
                print("true" if v.value else "false")
            elif isinstance(v, A.CharLit):
                print(repr(v.value))
            else:
                print(repr(v))
            return 0
    print(f"error: autodiff_cli: const {const_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _list_enums_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-enums in
    machine-readable JSON form.

    Output schema:
      {"enums": [{"name": "<name>", "variants": N,
                   "variant_names": ["<v1>", ...]}, ...]}
    Declaration order preserved.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    enums: list[dict] = []
    for it in prog.items:
        if isinstance(it, A.EnumDecl):
            variant_names = [v.name for v in it.variants]
            enums.append({
                "name": it.name,
                "variants": len(it.variants),
                "variant_names": variant_names,
            })
    print(json.dumps({"enums": enums}, sort_keys=True, indent=2))
    return 0


def _enum_variants_json(path: str, enum_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --enum-variants in
    machine-readable JSON form.

    Output schema:
      {"variants": [{"name": "<name>",
                      "payload_tys": ["<ty1>", "<ty2>", ...]}, ...]}
    Declaration order preserved. payload_tys is empty list for
    no-payload variants.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.EnumDecl) and it.name == enum_name:
            variants = [
                {"name": v.name,
                 "payload_tys": [_format_ty(t) for t in v.payload_tys]}
                for v in it.variants
            ]
            print(json.dumps({"variants": variants},
                              sort_keys=True, indent=2))
            return 0
    print(f"error: autodiff_cli: enum {enum_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _enum_variants(path: str, enum_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-enum variant
    inspection. Print variants of a specific enum with payload types.

    Output: one line per variant as:
      '<variant>' if no payload
      '<variant>(<ty1>, <ty2>, ...)' if tuple payload

    Declaration order preserved (matches enum tag-assignment order
    used by ABI / encoding).

    Exit 0 success; 1 if enum_name not found.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.EnumDecl) and it.name == enum_name:
            for v in it.variants:
                if v.payload_tys:
                    tys = ", ".join(_format_ty(t) for t in v.payload_tys)
                    print(f"{v.name}({tys})")
                else:
                    print(v.name)
            return 0
    print(f"error: autodiff_cli: enum {enum_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _list_type_aliases_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-type-aliases in
    machine-readable JSON form.

    Output schema:
      {"type_aliases": [{"name": "<name>", "target": "<ty>"}, ...]}
    Declaration order preserved.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    aliases = [
        {"name": it.name, "target": _format_ty(it.target)}
        for it in prog.items
        if isinstance(it, A.TypeAlias)
    ]
    print(json.dumps({"type_aliases": aliases},
                      sort_keys=True, indent=2))
    return 0


def _list_agents_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-agents in
    machine-readable JSON form.

    Output schema:
      {"agents": [{"name": "<name>", "methods": N,
                    "method_names": ["<m1>", ...]}, ...]}
    Declaration order preserved.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    agents: list[dict] = []
    for it in prog.items:
        if isinstance(it, A.AgentDecl):
            agents.append({
                "name": it.name,
                "methods": len(it.methods),
                "method_names": [m.name for m in it.methods],
            })
    print(json.dumps({"agents": agents}, sort_keys=True, indent=2))
    return 0


def _agent_methods_json(path: str, agent_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --agent-methods in
    machine-readable JSON form.

    Output schema:
      {"methods": [{"name": "<name>",
                     "params": ["<p1_ty>", ...],
                     "return_ty": "<ty>"}, ...]}
    Declaration order preserved.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        if ty is None:
            return "()"
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.AgentDecl) and it.name == agent_name:
            methods = [
                {"name": m.name,
                 "params": [_format_ty(p.ty) for p in m.params],
                 "return_ty": _format_ty(m.return_ty)}
                for m in it.methods
            ]
            print(json.dumps({"methods": methods},
                              sort_keys=True, indent=2))
            return 0
    print(f"error: autodiff_cli: agent {agent_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _impl_methods(path: str, target: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-ImplBlock method
    signature inspection.

    Output: one line per method as
        '<name>(<p1_ty>, <p2_ty>, ...) -> <ret_ty>'
    Declaration order preserved. If multiple impl blocks target the
    same type, methods from all of them are concatenated in
    declaration order.

    Per-item introspection family now spans 6 Item subclasses
    (struct-fields, const-value, enum-variants, agent-methods,
    type-alias-target, impl-methods).

    Exit 0 success; 1 if no impl block targets `target`.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        if ty is None:
            return "()"
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    found = False
    for it in prog.items:
        if isinstance(it, A.ImplBlock) and it.target == target:
            found = True
            for m in it.methods:
                params_str = ", ".join(_format_ty(p.ty) for p in m.params)
                ret_str = _format_ty(m.return_ty)
                print(f"{m.name}({params_str}) -> {ret_str}")
    if not found:
        print(f"error: autodiff_cli: no impl block found for "
              f"target {target!r} in {path}", file=sys.stderr)
        return 1
    return 0


def _impl_methods_json(path: str, target: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --impl-methods in
    machine-readable JSON form.

    Output schema:
      {"target": "<name>",
       "methods": [{"name": "<name>",
                     "params": ["<p1_ty>", ...],
                     "return_ty": "<ty>",
                     "trait": "<trait>" | null}, ...]}
    Declaration order preserved across multiple impl blocks for the
    same target. `trait` is populated when the method comes from a
    trait impl block.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        if ty is None:
            return "()"
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    methods: list[dict] = []
    found = False
    for it in prog.items:
        if isinstance(it, A.ImplBlock) and it.target == target:
            found = True
            for m in it.methods:
                methods.append({
                    "name": m.name,
                    "params": [_format_ty(p.ty) for p in m.params],
                    "return_ty": _format_ty(m.return_ty),
                    "trait": it.trait_name,
                })
    if not found:
        print(f"error: autodiff_cli: no impl block found for "
              f"target {target!r} in {path}", file=sys.stderr)
        return 1
    print(json.dumps({"target": target, "methods": methods},
                      sort_keys=True, indent=2))
    return 0


def _list_impls(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate top-level
    ImplBlock decls (inherent or trait impls) in a file.

    Output: one line per impl block as
        '<target> methods=N'              (inherent impl)
        '<trait> for <target> methods=N'  (trait impl)
    Declaration order preserved.

    Top-level enumeration octet + 1: fns / structs / modules / uses /
    consts / enums / type-aliases / agents + impls — closes the
    Item subclass enumeration to 9 axes.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for it in prog.items:
        if isinstance(it, A.ImplBlock):
            n = len(it.methods)
            if it.trait_name:
                print(f"{it.trait_name} for {it.target} methods={n}")
            else:
                print(f"{it.target} methods={n}")
    return 0


def _list_impls_json(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --list-impls in
    machine-readable JSON form.

    Output schema:
      {"impls": [{"target": "<name>",
                   "trait": "<name>" | null,
                   "methods": N,
                   "method_names": ["<m1>", ...]}, ...]}
    Declaration order preserved.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    impls: list[dict] = []
    for it in prog.items:
        if isinstance(it, A.ImplBlock):
            impls.append({
                "target": it.target,
                "trait": it.trait_name,
                "methods": len(it.methods),
                "method_names": [m.name for m in it.methods],
            })
    print(json.dumps({"impls": impls}, sort_keys=True, indent=2))
    return 0


def _type_alias_target_json(path: str, alias_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --type-alias-target in
    machine-readable JSON form.

    Output schema:
      {"name": "<name>", "target": "<ty>"}
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.TypeAlias) and it.name == alias_name:
            result = {"name": it.name, "target": _format_ty(it.target)}
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
    print(f"error: autodiff_cli: type alias {alias_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _type_alias_target(path: str, alias_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-TypeAlias target
    introspection. Print the target type a TypeAlias resolves to.

    Output: a single line with the formatted target type (e.g.,
    'i32' or 'D<f32>').

    Per-item introspection family now spans 5 Item subclasses
    (struct-fields, const-value, enum-variants, agent-methods,
    type-alias-target).

    Exit 0 success; 1 if alias_name not found.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.TypeAlias) and it.name == alias_name:
            print(_format_ty(it.target))
            return 0
    print(f"error: autodiff_cli: type alias {alias_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _agent_methods(path: str, agent_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: per-agent method
    signature inspection.

    Output: one line per method as
        '<name>(<p1_ty>, <p2_ty>, ...) -> <ret_ty>'
    Declaration order preserved.

    Per-item introspection trio + 1:
    - --struct-fields (StructDecl fields)
    - --const-value (ConstDecl literal value)
    - --enum-variants (EnumDecl variants + payload types)
    - --agent-methods (AgentDecl method signatures) — NEW

    Exit 0 success, 1 if agent_name not found.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        if ty is None:
            return "()"
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.AgentDecl) and it.name == agent_name:
            for m in it.methods:
                params_str = ", ".join(_format_ty(p.ty) for p in m.params)
                ret_str = _format_ty(m.return_ty)
                print(f"{m.name}({params_str}) -> {ret_str}")
            return 0
    print(f"error: autodiff_cli: agent {agent_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _list_agents(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate top-level
    AgentDecls (cognitive-architecture method bundles) in a file.

    Output: one line per agent as '<name> methods=N' (declaration
    order). Empty file → no output.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for it in prog.items:
        if isinstance(it, A.AgentDecl):
            print(f"{it.name} methods={len(it.methods)}")
    return 0


def _list_type_aliases(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate top-level
    TypeAlias decls in a file.

    Output: one line per alias as '<name> = <ty>' (declaration
    order).
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.TypeAlias):
            print(f"{it.name} = {_format_ty(it.target)}")
    return 0


def _list_enums(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate top-level
    EnumDecls in a file.

    Output: one line per enum as '<name> variants=N' (declaration
    order; not sorted — declaration order affects ABI).

    Extends the top-level item enumeration set to cover EnumDecl
    (joining FnDecl/StructDecl/ModBlock/UseDecl/ConstDecl).
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    for it in prog.items:
        if isinstance(it, A.EnumDecl):
            print(f"{it.name} variants={len(it.variants)}")
    return 0


def _list_consts(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate top-level
    ConstDecls in a file.

    Output: one line per constant as `<name>: <ty>` (declaration
    order, NOT sorted — declaration order matters for initialization
    sequencing).

    Companion to --list-fns / --list-structs / --list-modules /
    --list-uses. Closes the top-level item enumeration tetrad
    (fns + structs + modules + uses + consts).
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.ConstDecl):
            print(f"{it.name}: {_format_ty(it.ty)}")
    return 0


def _list_uses(path: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: enumerate `use` decls
    (imports) in a file.

    Output: one line per use-decl as dot-joined path segments:
        '<seg1>.<seg2>....<segN>'
    Sorted alphabetically for stable diff-friendly output.

    Use case: dependency inventory — what does this file depend on?
    Companion to --fn-callgraph (intra-file deps) and --list-modules
    (intra-file definitions); this covers cross-file references.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)
    uses: list[str] = []
    for it in prog.items:
        if isinstance(it, A.UseDecl):
            uses.append(".".join(it.path))
    for u in sorted(uses):
        print(u)
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


def _fn_signature(path: str, fn_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: print the full source-
    level signature of a fn as a single line.

    Output format:
        '[pub ][extern "C" ]fn <name>[<G1, G2>](p1: T1, p2: T2) -> R'
    or for void-returning fns:
        '[pub ][extern "C" ]fn <name>[<G1, G2>](p1: T1, p2: T2)'

    Per-item introspection family now spans 7 Item subclasses
    (struct-fields, const-value, enum-variants, agent-methods,
    type-alias-target, impl-methods, fn-signature).

    Exit 0 success; 1 if fn_name not found.
    """
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        if ty is None:
            return "()"
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.FnDecl) and it.name == fn_name:
            parts = []
            if it.is_pub:
                parts.append("pub")
            if it.is_extern:
                abi = it.extern_abi or "C"
                parts.append(f'extern "{abi}"')
            parts.append("fn")
            head = f"{' '.join(parts)} {it.name}"
            if it.generics:
                gs = ", ".join(g.name for g in it.generics)
                head += f"<{gs}>"
            params_str = ", ".join(
                f"{p.name}: {_format_ty(p.ty)}" for p in it.params
            )
            sig = f"{head}({params_str})"
            if it.return_ty is not None:
                sig += f" -> {_format_ty(it.return_ty)}"
            print(sig)
            return 0
    print(f"error: autodiff_cli: fn {fn_name!r} not found in {path}",
          file=sys.stderr)
    return 1


def _fn_signature_json(path: str, fn_name: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --fn-signature in
    machine-readable JSON form.

    Output schema:
      {"name": "<name>",
       "generics": [{"name": "<g>", "kind": "<kind>"}, ...],
       "params": [{"name": "<p>", "ty": "<ty>", "is_mut": bool}, ...],
       "return_ty": "<ty>" | null,
       "attrs": ["@pure", ...],
       "is_pub": bool,
       "is_extern": bool,
       "extern_abi": "<abi>" | null}

    Exit 0 success; 1 if fn_name not found.
    """
    import json
    src = _read_source(path)
    prog = _parse_or_exit(src, path)

    def _format_ty(ty) -> str:
        if ty is None:
            return None
        name = getattr(ty, "name", None)
        if name:
            return name
        base = getattr(ty, "base", None)
        args = getattr(ty, "args", None)
        base_str = base if isinstance(base, str) else (
            getattr(base, "name", None) if base else None
        )
        if base_str and args:
            args_strs = []
            for a in args:
                an = getattr(a, "name", None)
                args_strs.append(an if an else repr(a))
            return f"{base_str}<{', '.join(args_strs)}>"
        return repr(ty)

    for it in prog.items:
        if isinstance(it, A.FnDecl) and it.name == fn_name:
            result = {
                "name": it.name,
                "generics": [
                    {"name": g.name, "kind": g.kind}
                    for g in it.generics
                ],
                "params": [
                    {"name": p.name, "ty": _format_ty(p.ty),
                     "is_mut": p.is_mut}
                    for p in it.params
                ],
                "return_ty": _format_ty(it.return_ty),
                "attrs": list(it.attrs),
                "is_pub": it.is_pub,
                "is_extern": it.is_extern,
                "extern_abi": it.extern_abi,
            }
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
    print(f"error: autodiff_cli: fn {fn_name!r} not found in {path}",
          file=sys.stderr)
    return 1


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


def _changed_fns_json(path_a: str, path_b: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --changed-fns in
    machine-readable JSON form.

    Output schema:
      {"added":   [{"name": "<name>", "hash": "<64hex>"}, ...],
       "removed": [{"name": "<name>", "hash": "<64hex>"}, ...],
       "modified":[{"name": "<name>",
                     "old_hash": "<64hex>",
                     "new_hash": "<64hex>"}, ...],
       "n_changed": N}

    Unlike the text form (which uses 12-hex short hashes for human
    readability), the JSON variant uses FULL 64-hex hashes for
    storage / comparison consumption.

    Exit 0 if no changes; 1 if at least one fn changed (matches
    text-form rc semantics).
    """
    import json
    from .ast_hash import structural_hash
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
    added = [{"name": n, "hash": b_fns[n]}
             for n in sorted(b_names - a_names)]
    removed = [{"name": n, "hash": a_fns[n]}
               for n in sorted(a_names - b_names)]
    modified = [
        {"name": n, "old_hash": a_fns[n], "new_hash": b_fns[n]}
        for n in sorted(a_names & b_names)
        if a_fns[n] != b_fns[n]
    ]
    n_changed = len(added) + len(removed) + len(modified)
    print(json.dumps(
        {"added": added, "removed": removed,
         "modified": modified, "n_changed": n_changed},
        sort_keys=True, indent=2))
    return 0 if n_changed == 0 else 1


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


def _diff_program_hash_json(path_a: str, path_b: str) -> int:
    """Stage 59 follow-on / Tier 4 #13 polish: --diff-program-hash in
    machine-readable JSON form.

    Output schema:
      {"match": bool,
       "a_hash": "<64hex>",
       "b_hash": "<64hex>",
       "a_sig_hash": "<64hex>",
       "b_sig_hash": "<64hex>",
       "kind": "match" | "body-only" | "signature-change"}

    Always includes both program and signature hashes so consumers
    can do further analysis without re-invocation.

    rc=0 on match, 1 on differ (matches text-form semantics).
    """
    import json
    from .ast_hash import program_hash, program_signature_hash
    src_a = _read_source(path_a)
    src_b = _read_source(path_b)
    prog_a = _parse_or_exit(src_a, path_a)
    prog_b = _parse_or_exit(src_b, path_b)
    ha = program_hash(prog_a)
    hb = program_hash(prog_b)
    sig_a = program_signature_hash(prog_a)
    sig_b = program_signature_hash(prog_b)
    if ha == hb:
        kind = "match"
    elif sig_a == sig_b:
        kind = "body-only"
    else:
        kind = "signature-change"
    result = {
        "match": ha == hb,
        "a_hash": ha,
        "b_hash": hb,
        "a_sig_hash": sig_a,
        "b_sig_hash": sig_b,
        "kind": kind,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if ha == hb else 1


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

    if "--dump-ast-hashes-json" in flags:
        if len(args) < 1:
            print("usage: --dump-ast-hashes-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_dump_ast_hashes_json(args[0]))

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

    if "--diff-trace-json" in flags:
        if len(args) < 2:
            print("usage: --diff-trace-json <a.json> <b.json>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_diff_trace_json(args[0], args[1]))

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

    if "--diff-program-hash-json" in flags:
        if len(args) < 2:
            print("usage: --diff-program-hash-json <a.hx> <b.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_diff_program_hash_json(args[0], args[1]))

    if "--changed-fns" in flags:
        if len(args) < 2:
            print("usage: --changed-fns <a.hx> <b.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_changed_fns(args[0], args[1]))

    if "--changed-fns-json" in flags:
        if len(args) < 2:
            print("usage: --changed-fns-json <a.hx> <b.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_changed_fns_json(args[0], args[1]))

    if "--fn-sig-hash" in flags:
        if len(args) < 2:
            print("usage: --fn-sig-hash <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_sig_hash(args[0], args[1]))

    if "--fn-signature" in flags:
        if len(args) < 2:
            print("usage: --fn-signature <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_signature(args[0], args[1]))

    if "--fn-signature-json" in flags:
        if len(args) < 2:
            print("usage: --fn-signature-json <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_signature_json(args[0], args[1]))

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

    if "--list-uses" in flags:
        if len(args) < 1:
            print("usage: --list-uses <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_uses(args[0]))

    if "--list-consts" in flags:
        if len(args) < 1:
            print("usage: --list-consts <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_consts(args[0]))

    if "--list-enums" in flags:
        if len(args) < 1:
            print("usage: --list-enums <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_enums(args[0]))

    if "--list-type-aliases" in flags:
        if len(args) < 1:
            print("usage: --list-type-aliases <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_type_aliases(args[0]))

    if "--list-agents" in flags:
        if len(args) < 1:
            print("usage: --list-agents <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_agents(args[0]))

    if "--list-agents-json" in flags:
        if len(args) < 1:
            print("usage: --list-agents-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_agents_json(args[0]))

    if "--agent-methods" in flags:
        if len(args) < 2:
            print("usage: --agent-methods <file.hx> <agent_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_agent_methods(args[0], args[1]))

    if "--agent-methods-json" in flags:
        if len(args) < 2:
            print("usage: --agent-methods-json <file.hx> <agent_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_agent_methods_json(args[0], args[1]))

    if "--list-impls" in flags:
        if len(args) < 1:
            print("usage: --list-impls <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_impls(args[0]))

    if "--list-impls-json" in flags:
        if len(args) < 1:
            print("usage: --list-impls-json <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_impls_json(args[0]))

    if "--impl-methods" in flags:
        if len(args) < 2:
            print("usage: --impl-methods <file.hx> <target>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_impl_methods(args[0], args[1]))

    if "--impl-methods-json" in flags:
        if len(args) < 2:
            print("usage: --impl-methods-json <file.hx> <target>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_impl_methods_json(args[0], args[1]))

    if "--type-alias-target" in flags:
        if len(args) < 2:
            print("usage: --type-alias-target <file.hx> <alias_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_type_alias_target(args[0], args[1]))

    if "--type-alias-target-json" in flags:
        if len(args) < 2:
            print("usage: --type-alias-target-json <file.hx> <alias_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_type_alias_target_json(args[0], args[1]))

    if "--list-type-aliases-json" in flags:
        if len(args) < 1:
            print("usage: --list-type-aliases-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_type_aliases_json(args[0]))

    if "--enum-variants" in flags:
        if len(args) < 2:
            print("usage: --enum-variants <file.hx> <enum_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_enum_variants(args[0], args[1]))

    if "--enum-variants-json" in flags:
        if len(args) < 2:
            print("usage: --enum-variants-json <file.hx> <enum_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_enum_variants_json(args[0], args[1]))

    if "--list-enums-json" in flags:
        if len(args) < 1:
            print("usage: --list-enums-json <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_enums_json(args[0]))

    if "--list-consts-json" in flags:
        if len(args) < 1:
            print("usage: --list-consts-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_consts_json(args[0]))

    if "--const-value" in flags:
        if len(args) < 2:
            print("usage: --const-value <file.hx> <const_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_const_value(args[0], args[1]))

    if "--const-value-json" in flags:
        if len(args) < 2:
            print("usage: --const-value-json <file.hx> <const_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_const_value_json(args[0], args[1]))

    if "--list-uses-json" in flags:
        if len(args) < 1:
            print("usage: --list-uses-json <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_uses_json(args[0]))

    if "--struct-fields" in flags:
        if len(args) < 2:
            print("usage: --struct-fields <file.hx> <struct_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_struct_fields(args[0], args[1]))

    if "--struct-fields-json" in flags:
        if len(args) < 2:
            print("usage: --struct-fields-json <file.hx> <struct_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_struct_fields_json(args[0], args[1]))

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

    if "--fn-callgraph-json" in flags:
        if len(args) < 2:
            print("usage: --fn-callgraph-json <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph_json(args[0], args[1]))

    if "--fn-body-stats" in flags:
        if len(args) < 2:
            print("usage: --fn-body-stats <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_body_stats(args[0], args[1]))

    if "--fn-body-stats-json" in flags:
        if len(args) < 2:
            print("usage: --fn-body-stats-json <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_body_stats_json(args[0], args[1]))

    if "--fn-body-stats-all" in flags:
        if len(args) < 1:
            print("usage: --fn-body-stats-all <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_body_stats_all(args[0]))

    if "--fn-body-stats-summary" in flags:
        if len(args) < 1:
            print("usage: --fn-body-stats-summary <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_body_stats_summary(args[0]))

    if "--fn-body-stats-rank" in flags:
        if len(args) < 3:
            print("usage: --fn-body-stats-rank <file.hx> <metric> <top_n>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_body_stats_rank(args[0], args[1], args[2]))

    if "--fn-callers" in flags:
        if len(args) < 2:
            print("usage: --fn-callers <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callers(args[0], args[1]))

    if "--fn-callers-json" in flags:
        if len(args) < 2:
            print("usage: --fn-callers-json <file.hx> <fn_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callers_json(args[0], args[1]))

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

    if "--fn-reachable-from-json" in flags:
        if len(args) < 2:
            print("usage: --fn-reachable-from-json <file.hx> <entry_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_reachable_from_json(args[0], args[1]))

    if "--fn-reachable-to" in flags:
        if len(args) < 2:
            print("usage: --fn-reachable-to <file.hx> <target_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_reachable_to(args[0], args[1]))

    if "--fn-reachable-to-json" in flags:
        if len(args) < 2:
            print("usage: --fn-reachable-to-json <file.hx> <target_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_reachable_to_json(args[0], args[1]))

    if "--fn-call-stats" in flags:
        if len(args) < 1:
            print("usage: --fn-call-stats <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_call_stats(args[0]))

    if "--fn-callgraph-depth" in flags:
        if len(args) < 2:
            print("usage: --fn-callgraph-depth <file.hx> <entry_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph_depth(args[0], args[1]))

    if "--fn-callgraph-depth-all" in flags:
        if len(args) < 1:
            print("usage: --fn-callgraph-depth-all <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph_depth_all(args[0]))

    if "--fn-topo-sort" in flags:
        if len(args) < 1:
            print("usage: --fn-topo-sort <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_topo_sort(args[0]))

    if "--fn-isolated" in flags:
        if len(args) < 1:
            print("usage: --fn-isolated <file.hx>", file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_isolated(args[0]))

    if "--fn-call-path" in flags:
        if len(args) < 3:
            print("usage: --fn-call-path <file.hx> <from_fn> <to_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_call_path(args[0], args[1], args[2]))

    if "--fn-distance" in flags:
        if len(args) < 3:
            print("usage: --fn-distance <file.hx> <from_fn> <to_fn>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_distance(args[0], args[1], args[2]))

    if "--fn-distance-matrix" in flags:
        if len(args) < 1:
            print("usage: --fn-distance-matrix <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_distance_matrix(args[0]))

    if "--fn-callgraph-summary" in flags:
        if len(args) < 1:
            print("usage: --fn-callgraph-summary <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph_summary(args[0]))

    if "--fn-callgraph-dot" in flags:
        if len(args) < 1:
            print("usage: --fn-callgraph-dot <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph_dot(args[0]))

    if "--fn-callgraph-mermaid" in flags:
        if len(args) < 1:
            print("usage: --fn-callgraph-mermaid <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_fn_callgraph_mermaid(args[0]))

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

    if "--list-fns-by-attr-json" in flags:
        if len(args) < 2:
            print("usage: --list-fns-by-attr-json <file.hx> <attr>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_list_fns_by_attr_json(args[0], args[1]))

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

    if "--module-hash-json" in flags:
        if len(args) < 2:
            print("usage: --module-hash-json <file.hx> <module_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_module_hash_json_cli(args[0], args[1]))

    if "--module-stats" in flags:
        if len(args) < 2:
            print("usage: --module-stats <file.hx> <module_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_module_stats(args[0], args[1]))

    if "--module-stats-json" in flags:
        if len(args) < 2:
            print("usage: --module-stats-json <file.hx> <module_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_module_stats_json(args[0], args[1]))

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

    if "--validate-trace-attrs-json" in flags:
        if len(args) < 1:
            print("usage: --validate-trace-attrs-json <file.hx>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_validate_trace_attrs_json(args[0]))

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

    if "--pytree-shape-json" in flags:
        if len(args) < 2:
            print("usage: --pytree-shape-json <file.hx> <struct_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_pytree_shape_json(args[0], args[1]))

    if "--pytree-leaf-paths-json" in flags:
        if len(args) < 2:
            print("usage: --pytree-leaf-paths-json <file.hx> <struct_name>",
                  file=sys.stderr)
            sys.exit(2)
        sys.exit(_pytree_leaf_paths_json(args[0], args[1]))

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
