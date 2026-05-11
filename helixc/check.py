"""
helixc/check.py — `python -m helixc.check <file.hx>` developer CLI.

Stage 23: full flag dispatch.

Pipeline:
  1. Lex + parse
  2. Typecheck (with did-you-mean suggestions)
  3. Totality (structural recursion; @partial fns skipped)
  4. Lower to Tensor IR (--emit-ir / --emit-asm / --emit-ptx)
  5. Codegen to x86_64 / PTX (--emit-asm / --emit-ptx / -o)
  6. Doc extraction (--doc)

Exit codes:
  0 = clean
  1 = compile error
  2 = bad invocation (missing file, unknown flag)

Flags:
  --stdlib              Bundle helixc/stdlib/transcendentals.hx
  --hash                Print structural hash per top-level fn
  --hash-cons           Dedup AST nodes, print rewrite count
  --strict              Fail if totality fails
  --check-only          Stop after typecheck + totality (no IR/codegen)
  --emit-ast            Print AST and exit
  --emit-ir             Print Tensor IR and exit
  --emit-asm            Print x86_64 hex/textual disassembly and exit
  --emit-ptx            Print PTX kernels and exit
  --doc                 Extract /// doc comments to markdown and exit
  -O0 / -O1 / -O2 / -O3
                        Optimization level (0=none, 1=fdce+fold,
                        2=+cse+dce, 3=alias of -O2 until an aggressive
                        layer lands). Default -O1.
  -o <path>             Write ELF output to <path> instead of default
  -l <libname>          Mark <libname> as external (FFI prerequisite)
  -W<flag>              Warning policy (e.g. -Wdeprecated, -Wdeprecated=error)
  --no-color            Disable ANSI escapes (also: NO_COLOR env)
  --color               Force ANSI escapes on
  -h / --help           Show this help

Examples:
    python -m helixc.check loss.hx
    python -m helixc.check --emit-ir loss.hx
    python -m helixc.check --check-only --strict loss.hx
    python -m helixc.check -O2 -o loss.bin loss.hx
    python -m helixc.check -l m -l c loss.hx
    python -m helixc.check --doc loss.hx > loss.md

License: Apache 2.0
"""

from __future__ import annotations

import sys
import os

from .frontend.parser import parse, ParseError
from .frontend.typecheck import typecheck
from .frontend.totality import check_totality
from .frontend.ast_hash import structural_hash, short_hash
from .frontend.hash_cons import hash_cons
from .frontend import ast_nodes as A
from .frontend import diagnostics as diag


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
class CliArgs:
    """Parsed CLI arguments. Public attributes match flag names."""
    def __init__(self):
        self.path: str | None = None
        self.flags: set[str] = set()
        self.opt_level: int = 1
        self.output: str | None = None
        self.libs: list[str] = []
        self.warnings: dict[str, str] = {}  # e.g. {"deprecated": "warn" or "error"}
        self.color: bool | None = None  # None = auto


_KNOWN_LONG_FLAGS = frozenset({
    "--stdlib", "--hash", "--hash-cons", "--strict",
    "--check-only", "--emit-ast", "--emit-ir", "--emit-asm",
    "--emit-ptx", "--doc", "--help", "--no-color", "--color",
    "-h",
})


def parse_args(argv: list[str]) -> tuple[CliArgs, list[str]]:
    """Parse argv into CliArgs. Returns (args, errors). errors is a
    list of human-readable strings (empty on success)."""
    a = CliArgs()
    errors: list[str] = []
    i = 0
    n = len(argv)
    positional: list[str] = []
    while i < n:
        tok = argv[i]
        if tok in ("-h", "--help"):
            a.flags.add("--help")
            i += 1
        elif tok == "--no-color":
            a.color = False
            i += 1
        elif tok == "--color":
            a.color = True
            i += 1
        elif tok in _KNOWN_LONG_FLAGS:
            a.flags.add(tok)
            i += 1
        elif tok.startswith("-O") and len(tok) == 3 and tok[2].isdigit():
            lvl = int(tok[2])
            if lvl < 0 or lvl > 3:
                errors.append(f"unknown opt level: {tok}")
            else:
                a.opt_level = lvl
            i += 1
        elif tok == "-o":
            if i + 1 >= n:
                errors.append("-o requires an argument")
                i += 1
            else:
                a.output = argv[i + 1]
                i += 2
        elif tok == "-l":
            if i + 1 >= n:
                errors.append("-l requires an argument")
                i += 1
            else:
                a.libs.append(argv[i + 1])
                i += 2
        elif tok.startswith("-l") and len(tok) > 2:
            a.libs.append(tok[2:])
            i += 1
        elif tok.startswith("-W"):
            body = tok[2:]
            if "=" in body:
                name, val = body.split("=", 1)
                a.warnings[name] = val
            else:
                a.warnings[body] = "warn"
            i += 1
        elif tok.startswith("--") or tok.startswith("-"):
            errors.append(f"unknown flag: {tok}")
            i += 1
        else:
            positional.append(tok)
            i += 1
    if positional:
        a.path = positional[0]
        if len(positional) > 1:
            errors.append(f"unexpected extra arg(s): {positional[1:]}")
    return a, errors


def _print_help():
    print(__doc__.strip())


# ----------------------------------------------------------------------
# Doc extraction
# ----------------------------------------------------------------------
def extract_doc_comments(src: str) -> str:
    """Stage 23 --doc: scan source for `///` comments, group with the
    following fn/struct/enum decl, emit markdown.

    A doc-comment block is a run of consecutive `///` lines (possibly
    indented). The block attaches to the next non-comment, non-blank
    line; we extract the symbol name from `fn NAME`, `struct NAME`,
    `enum NAME`, `trait NAME`."""
    out_lines: list[str] = ["# Doc comments\n"]
    lines = src.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("///"):
            block: list[str] = []
            while i < n and lines[i].strip().startswith("///"):
                # Trim leading "///" and one optional space
                txt = lines[i].strip()[3:]
                if txt.startswith(" "):
                    txt = txt[1:]
                block.append(txt)
                i += 1
            # Skip blank lines
            while i < n and not lines[i].strip():
                i += 1
            # Attach to next decl-line
            sym = "<unattached>"
            kind = "?"
            if i < n:
                following = lines[i].lstrip()
                for k in ("fn", "struct", "enum", "trait", "impl", "mod"):
                    if following.startswith(k + " "):
                        rest = following[len(k) + 1:]
                        # Extract identifier up to '(' '<' '{' or whitespace
                        name = ""
                        for ch in rest:
                            if ch.isalnum() or ch == "_":
                                name += ch
                            else:
                                break
                        sym = name or "<anon>"
                        kind = k
                        break
            out_lines.append(f"## `{kind} {sym}`\n")
            for b in block:
                out_lines.append(b)
            out_lines.append("")
        else:
            i += 1
    return "\n".join(out_lines)


# ----------------------------------------------------------------------
# AD warning drain (Audit 28.8 cycle 2 C2-1)
# ----------------------------------------------------------------------
def _drain_ad_warnings(a: "CliArgs") -> int:
    """Drain the AD-warning channel and emit diagnostics to stderr.

    Returns 1 if `-Wad=error` was set AND any warnings were drained,
    else 0. Must be called on every code path that exits successfully
    from `main()` so that B13 widening warnings emitted during typecheck
    are surfaced even when the user runs with no `--emit-*` / `-o` /
    or `--check-only`.

    The drain itself ALWAYS clears `_DIFF_WARNINGS` (via `take_diff_
    warnings`) — even when the policy is "warn" rather than "error",
    so a subsequent compile in the same process starts clean.
    """
    from .frontend.autodiff import take_diff_warnings
    ad_warnings = take_diff_warnings()
    if not ad_warnings:
        return 0
    ad_policy = a.warnings.get("ad", "warn")
    label = "ERROR" if ad_policy == "error" else "warning"
    print(f"   ad:        {len(ad_warnings)} {label}(s)")
    for w in ad_warnings:
        print(f"     helixc: {w}", file=sys.stderr)
    if ad_policy == "error":
        return 1
    return 0


def _emit_env_error(msg: str) -> None:
    """Audit 28.8 cycle 9 C8-2: print a user-environment error with a
    single `helixc:` prefix. Strips an already-present `helixc:` prefix
    from `msg` so callees that raise with the prefix already formatted
    (e.g. parser.py:1587's strict-stdlib FileNotFoundError) don't
    double-print as `helixc: helixc: ...`."""
    text = msg
    if text.lstrip().startswith("helixc:"):
        text = text.lstrip()[len("helixc:"):].lstrip()
    print(f"helixc: {text}", file=sys.stderr)


# ----------------------------------------------------------------------
# Main dispatch
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Outer dispatch.

    Audit 28.8 cycle 2 C2-1: wraps `_main_inner` so the AD warning drain
    runs on EVERY exit (success or error, after the entry banner is
    printed). Pre-fix the drain lived inside the lowering branch and was
    bypassed on default-no-emit, `--check-only`, and error returns —
    silently dropping B13 widening warnings emitted during typecheck.
    """
    # Clear any stale AD warnings from a prior compilation in the same
    # process before we start. Without this, the module-level
    # `_DIFF_WARNINGS` list accumulates across compiles, mis-attributing
    # diagnostics to the wrong file when check.main() is invoked twice.
    from .frontend.autodiff import take_diff_warnings as _drain_ad_init
    _drain_ad_init()
    a_holder: list["CliArgs"] = []
    rc = 1
    # Audit 28.8 cycle 3 C3-3: wrap _main_inner in try/finally so
    # exception exits ALSO trigger the AD-warning drain and present a
    # clean error message instead of a raw Python traceback. Without
    # this, a typecheck/struct_mono/lower/codegen bug leaks both the
    # traceback AND the accumulated `_DIFF_WARNINGS` (since the drain
    # at the bottom is never reached).
    try:
        rc = _main_inner(argv, a_holder)
    # Audit 28.8 cycle 5 C4-6 / MEDIUM: distinguish user-environment
    # errors (file I/O, encoding) from genuine compiler bugs. Pre-fix
    # the broad `except Exception` printed "this is a compiler bug —
    # please file an issue" for FileNotFound, UnicodeDecodeError, etc.
    # — which are NOT compiler bugs. Now: env errors get a clean
    # `helixc:` message with rc=2 (config / invocation error); only
    # genuine pipeline-internal exceptions get the "compiler bug"
    # tagline.
    #
    # Audit 28.8 cycle 9 C8-2: outer-arm message strips an already-
    # present `helixc:` prefix to avoid double-printing when a callee
    # raised the error with the prefix already formatted in its
    # message.
    except (FileNotFoundError, PermissionError, IsADirectoryError,
            NotADirectoryError) as e:
        _emit_env_error(str(e))
        rc = 2
    except UnicodeDecodeError as e:
        _emit_env_error(f"encoding error reading source: {e}")
        rc = 2
    except Exception as e:
        # Everything else (AttributeError, KeyError, IndexError,
        # AssertionError, TypeError, RuntimeError, ValueError, etc.)
        # is a genuine internal-error candidate.
        print(
            f"helixc: internal error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        print(
            "helixc: this is a compiler bug — please file an issue.",
            file=sys.stderr,
        )
        rc = 1
    finally:
        # Audit 28.8 cycle 5 C4-6: wrap the drain itself so a drain
        # failure doesn't mask the primary failure. Pre-fix, if
        # `_drain_ad_warnings` raised in the finally, the new exception
        # propagated up as a raw traceback masking the original.
        try:
            if a_holder:
                drain_rc = _drain_ad_warnings(a_holder[0])
                if drain_rc != 0 and rc == 0:
                    rc = drain_rc
            else:
                # No CliArgs yet — drain quietly to keep state hygienic.
                _drain_ad_init()
        except Exception as drain_e:
            print(
                f"helixc: warning: AD-warning drain failed: "
                f"{type(drain_e).__name__}: {drain_e}",
                file=sys.stderr,
            )
    return rc


def _main_inner(argv: list[str] | None,
                a_holder: list["CliArgs"]) -> int:
    """Inner pipeline. Pushes the parsed CliArgs into `a_holder` (a
    single-element list passed by the caller) as soon as parsing
    succeeds, so the outer wrapper can run the AD-warning drain even
    when this function returns early. Pre-banner short-circuits (`--help`,
    `--doc`, bad invocation) don't push anything — those exits have no
    `_DIFF_WARNINGS` to drain because typecheck never ran."""
    argv = list(argv if argv is not None else sys.argv[1:])
    a, errs = parse_args(argv)
    if "--help" in a.flags:
        _print_help()
        return 0
    if errs:
        for e in errs:
            print(f"helixc: {e}", file=sys.stderr)
        return 2
    if a.path is None:
        _print_help()
        return 2
    path = a.path
    if not os.path.exists(path):
        print(f"helixc: file not found: {path}", file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # Doc extraction is a separate mode: parse not required.
    if "--doc" in a.flags:
        print(extract_doc_comments(src))
        return 0

    print(f"-- helixc-check: {path}")
    # Audit 28.8 cycle 2 C2-1: register CliArgs so the outer wrapper
    # can drain AD warnings on ANY return below (including error paths).
    a_holder.append(a)

    # 1. Parse
    try:
        prog = parse(src, include_stdlib=("--stdlib" in a.flags))
    except ParseError as e:
        rendered = e.render(source=src, filename=path, color=a.color)
        print("PARSE ERROR:", file=sys.stderr)
        for line in rendered.splitlines():
            print(f"  {line}", file=sys.stderr)
        return 1
    fn_count = sum(1 for it in prog.items if isinstance(it, A.FnDecl))
    print(f"   parse:    OK  ({fn_count} fns, {len(prog.items)} items)")

    # --emit-ast (early exit before typecheck for debugging parser)
    if "--emit-ast" in a.flags:
        for it in prog.items:
            if isinstance(it, A.FnDecl):
                print(f"fn {it.name}({len(it.params)} params) -> ...")
            elif isinstance(it, A.StructDecl):
                print(f"struct {it.name}({len(it.fields)} fields)")
            else:
                print(f"{type(it).__name__}")
        return 0

    # 2. Typecheck
    tc_errs = typecheck(prog)
    if tc_errs:
        print(f"   typecheck: {len(tc_errs)} ERRORS")
        for e in tc_errs[:20]:
            rendered = e.render(source=src, filename=path, color=a.color) \
                if hasattr(e, "render") else str(e)
            for line in rendered.splitlines():
                print(f"     {line}")
        if len(tc_errs) > 20:
            print(f"     ... and {len(tc_errs) - 20} more")
        return 1
    print(f"   typecheck: OK")

    # 2.5 Stage 28 — parametric-struct monomorphization (Audit 28.8
    # A3/B1/C1-M2). Surfaces arity-mismatch diagnostics + appends the
    # mono'd StructDecls so downstream passes (lowering, codegen) can
    # find them by mangled name. Doesn't replace typecheck's lookup
    # (which goes through `_resolve_type` -> `mangle_struct` and
    # produces `TyStruct(mangled)` directly).
    from .frontend.struct_mono import monomorphize_structs
    prog, sm_diags = monomorphize_structs(prog)
    if sm_diags:
        print(f"   struct-mono: {len(sm_diags)} ERROR(s)")
        for d in sm_diags:
            print(f"     {d}")
        return 1

    # 2.6 Audit 28.8 cycle 2 B:C7 — flatten impls so trap 74002
    # (duplicate method name across distinct structs) is reachable
    # from the surface tool. Pre-fix `flatten_impls` was only invoked
    # inside the backend, so users iterating with `helixc check
    # foo.hx` never saw the diagnostic until they tried to emit a
    # binary. Catch the structured DuplicateMethodError and turn it
    # into a clean per-line message; any other exception bubbles
    # (it's an internal compiler issue).
    from .frontend.flatten_impls import (
        flatten_impls, DuplicateMethodError,
    )
    try:
        flatten_impls(prog)
    except DuplicateMethodError as e:
        print(f"   impl-flatten: ERROR")
        print(f"     {e}")
        return 1

    # 3. Totality
    fails = check_totality(prog)
    if fails:
        print(f"   totality:  {len(fails)} fn(s) NOT proven total")
        for name, reason in fails:
            print(f"     {name}: {reason}")
        if "--strict" in a.flags:
            return 1
    else:
        print(f"   totality:  OK")

    # Stage 28.7: @deprecated pass. Runs the Python-side walker and
    # collects warnings; the -Wdeprecated=error flag promotes them.
    from .frontend.deprecated_pass import emit_warnings
    deprecate_policy = a.warnings.get("deprecated", "warn")
    warnings = emit_warnings(prog)
    if warnings:
        label = "ERROR" if deprecate_policy == "error" else "warning"
        print(f"   deprecated: {len(warnings)} {label}(s)")
        for w in warnings:
            print(f"     {w}")
        if deprecate_policy == "error":
            return 1

    # Stage 25: @trace validation pass (Audit 28.8 A7).
    # `validate_trace_attrs` rejects @trace on extern "C" fns (no body
    # to instrument). Codegen-side wiring (IR ops TRACE_ENTRY /
    # TRACE_EXIT) is handled in ir/lower_ast.py + backend/x86_64.py.
    from .frontend.trace_pass import validate_trace_attrs
    trace_diags = validate_trace_attrs(prog)
    if trace_diags:
        print(f"   trace:     {len(trace_diags)} ERROR(s)")
        for d in trace_diags:
            print(f"     {d}")
        return 1

    # Stage 28.5: panic / unwind validation passes (Audit 28.8 A1).
    # `validate_panic_args` enforces single-string-literal arg shape;
    # `validate_unwind` rejects @unwind (trap 28502 reserved).
    # Non-empty diagnostics fail the build (returning 1) because they
    # represent real malformed source — codegen would either emit
    # garbage or raise at lowering time.
    from .frontend.panic_pass import (
        validate_panic_args, validate_unwind,
    )
    panic_diags = validate_panic_args(prog)
    unwind_diags = validate_unwind(prog)
    if panic_diags:
        print(f"   panic:     {len(panic_diags)} ERROR(s)")
        for d in panic_diags:
            print(f"     {d}")
    if unwind_diags:
        print(f"   unwind:    {len(unwind_diags)} ERROR(s)")
        for d in unwind_diags:
            print(f"     {d}")
    if panic_diags or unwind_diags:
        return 1

    # Stage 28.6: unsafe capability gate (Audit 28.8 A2).
    # `check_unsafe_ops` flags raw-pointer ops (Unary deref) outside
    # any enclosing UnsafeBlock with trap 28601. Empty list = clean.
    from .frontend.unsafe_pass import check_unsafe_ops
    unsafe_diags = check_unsafe_ops(prog)
    if unsafe_diags:
        print(f"   unsafe:    {len(unsafe_diags)} ERROR(s)")
        for d in unsafe_diags:
            print(f"     {d}")
        return 1

    # Stage 27: @autotune validation (Audit 28.8 A12).
    # `validate_autotune_prog` runs `validate_autotune` over every
    # autotuned fn and surfaces diagnostics for malformed attrs
    # (non-int values, missing `=`), oversized cross-product, missing
    # @kernel pairing. Pre-fix this pass existed but was never invoked
    # by check.py — so a user @autotune(BS: [16, "fast", 32]) saw
    # neither the typo NOR the empty-params secondary diagnostic.
    from .frontend.autotune import validate_autotune_prog
    autotune_diags = validate_autotune_prog(prog)
    if autotune_diags:
        print(f"   autotune:  {len(autotune_diags)} ERROR(s)")
        for d in autotune_diags:
            print(f"     {d}")
        return 1


    # 4. Optional hash dump
    if "--hash" in a.flags:
        print("   hashes:")
        for it in prog.items:
            if isinstance(it, A.FnDecl):
                print(f"     {it.name:<40} {short_hash(structural_hash(it))}")

    # 4.5 Optional hash-cons
    if "--hash-cons" in a.flags:
        n_shared = hash_cons(prog)
        print(f"   hash-cons: {n_shared} AST node(s) deduped")

    # --check-only short-circuit: stop here.
    if "--check-only" in a.flags:
        print("-- clean (check-only)")
        return 0

    # 5. Lower + (optional) optimization passes
    if any(f in a.flags for f in ("--emit-ir", "--emit-asm", "--emit-ptx")) \
            or a.output is not None:
        from .ir.lower_ast import lower
        from .ir.passes.fdce import fdce_module
        from .ir.passes.const_fold import fold_module
        from .ir.passes.cse import cse_module
        from .ir.passes.dce import dce_module
        # Audit 28.8 B5: invoke grad_pass before lowering so any
        # `grad(loss)` calls are rewritten to symbolic gradients (and
        # the AD passes' diagnostics accumulate in _DIFF_WARNINGS).
        # We import lazily so check.py's start-up cost stays low for
        # programs that don't use grad().
        from .frontend.grad_pass import grad_pass
        grad_pass(prog)
        mod = lower(prog)
        # Optimization pipeline (Audit 28.8 A10):
        #   -O0          — no opts
        #   -O1 (default)— fdce + const_fold
        #   -O2          — -O1 + cse + dce
        #   -O3          — -O2 (no aggressive layer yet; surface here so
        #                  future passes have a hook). Documented identical
        #                  to -O2 until a Phase-0+ aggressive layer exists.
        #
        # The pre-fix code ran ONLY fdce at -O1+ and stubbed -O2 with an
        # empty try/import-pass placeholder — so users invoking `-O2`
        # got the SAME IR as `-O1`, despite the help text promising
        # +cse+dce. The fix wires the real passes; the placeholder is
        # removed.
        if a.opt_level >= 1:
            fdce_module(mod)
            fold_module(mod)
        if a.opt_level >= 2:
            cse_module(mod)
            dce_module(mod)
        if a.opt_level >= 3:
            # No aggressive layer yet — kept distinct from -O2 so a
            # future pass can wire in without re-shaping the dispatch.
            pass

        # Audit 28.8 cycle 2 C2-1: the AD-warning drain was moved out
        # of this branch into the outer `main()` wrapper. Pre-fix, the
        # drain only ran when an `--emit-*` flag or `-o` was set, so
        # default-no-emit and `--check-only` silently lost the B13
        # widening warnings emitted during typecheck. Drain is now
        # universal — see `_drain_ad_warnings` and the `main` wrapper.

    if "--emit-ir" in a.flags:
        print("   ir:")
        for fn in mod.functions.values():
            print(f"     fn {fn.name}:")
            for blk in fn.blocks:
                print(f"       block {blk.id}:")
                for op in blk.ops:
                    operands = ",".join(str(o.id) for o in op.operands)
                    results = ",".join(str(r.id) for r in op.results)
                    attrs_str = (" " + str(dict(op.attrs))) if op.attrs else ""
                    print(f"         {op.kind.name} ({operands}) -> ({results}){attrs_str}")
        return 0

    if "--emit-asm" in a.flags:
        from .backend.x86_64 import compile_module_to_elf
        # Audit 28.8 A9: wrap backend call in try/except so internal
        # compiler errors render as `helixc: internal error: ...`
        # instead of leaking Python tracebacks. Mirrors the existing
        # --emit-ptx error-path pattern (line 422-427).
        try:
            elf = compile_module_to_elf(mod)
        except Exception as e:
            print(
                f"helixc: internal error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            print(
                "helixc: this is a compiler bug — please file an issue.",
                file=sys.stderr,
            )
            return 1
        # Phase-0: ELF hex dump. A real disassembler would use objdump.
        print(f"   asm: {len(elf)} bytes of ELF (use `objdump -d` for asm)")
        for i in range(0, len(elf), 16):
            row = elf[i:i + 16]
            hex_row = " ".join(f"{b:02x}" for b in row)
            print(f"     {i:08x}  {hex_row}")
        return 0

    if "--emit-ptx" in a.flags:
        from .ir import tile_ir as ti
        from .backend.ptx import emit_ptx
        # PTX requires a TileModule. Phase-0: report empty if no kernels.
        kernel_count = sum(1 for it in prog.items
                           if isinstance(it, A.FnDecl) and getattr(it, "is_kernel", False))
        if kernel_count == 0:
            print("   ptx: no @kernel fns in program")
            return 0
        tile_mod = ti.TileModule()  # placeholder
        try:
            ptx = emit_ptx(tile_mod)
            print(ptx)
        except Exception as e:
            print(f"   ptx: backend error: {e}", file=sys.stderr)
            return 1
        return 0

    # 6. Codegen to ELF (when -o)
    if a.output is not None:
        from .backend.x86_64 import compile_module_to_elf
        # Audit 28.8 A9: wrap backend call so internal codegen errors
        # render cleanly. Also catch OSError on the file write so
        # permission / disk-full failures surface as a real diagnostic,
        # not a partial-file + traceback.
        try:
            elf = compile_module_to_elf(mod)
        except Exception as e:
            print(
                f"helixc: internal error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            print(
                "helixc: this is a compiler bug — please file an issue.",
                file=sys.stderr,
            )
            return 1
        try:
            with open(a.output, "wb") as f:
                f.write(elf)
        except OSError as e:
            print(
                f"helixc: cannot write output {a.output!r}: {e}",
                file=sys.stderr,
            )
            return 1
        print(f"   codegen:   OK  -> {a.output} ({len(elf)} bytes)")
        if a.libs:
            # Phase-0: libs are recorded but actual linking is stage-15.5
            # FFI's job. Surface them for user visibility.
            print(f"   libs:     {', '.join(a.libs)}")

    print("-- clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
