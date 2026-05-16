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
  --stdlib              Bundle stdlib (default; kept for compatibility)
  --no-stdlib           Do not bundle helixc/stdlib/*.hx
  --hash                Print structural hash per top-level fn
  --hash-cons           Dedup AST nodes, print rewrite count
  --strict              Fail if totality fails
  --check-only          Stop after typecheck + totality (no IR/codegen)
  --emit-ast            Print AST and exit
  --emit-ir             Print Tensor IR and exit
  --emit-proof-obligations
                        Print Stage 31 proof-obligation JSON and exit
  --emit-asm            Print x86_64 hex/textual disassembly and exit
  --emit-ptx            Print PTX kernels and exit
  --doc                 Extract /// doc comments to markdown and exit
  -O0 / -O1 / -O2 / -O3
                        Optimization level (0=none, 1=fold+cse+dce+fdce
                        for host IR/ELF; PTX skips DCE/FDCE so emitted
                        kernel text stays inspectable. 2/3 currently alias
                        -O1 until stronger layers land). Default -O1.
  --no-opt              Synonym for -O0 (parity with backend CLIs).
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

import dataclasses
import json
import hashlib
import sys
import os
import tempfile

from .frontend.lexer import LexError
from .frontend import parser as parser_mod
from .frontend.parser import parse, ParseError
from .frontend.typecheck import typecheck, typecheck_with_proof_artifacts
from .frontend.totality import check_totality
from .frontend.ast_hash import structural_hash, short_hash
from .frontend.hash_cons import hash_cons
from .frontend import ast_nodes as A
from .frontend import diagnostics as diag


PROOF_SCHEMA = "helix.proof_obligations.v0"


def _called_fn_names(value: object) -> set[str]:
    names: set[str] = set()
    seen: set[int] = set()

    def visit(node: object) -> None:
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
            return
        oid = id(node)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(node, A.Call) and isinstance(node.callee, A.Name):
            names.add(node.callee.name)
        if dataclasses.is_dataclass(node):
            for field in dataclasses.fields(node):
                visit(getattr(node, field.name))

    visit(value)
    return names


def _kernel_reachable_program(prog: A.Program) -> A.Program:
    fn_by_name = {
        it.name: it for it in prog.items if isinstance(it, A.FnDecl)
    }
    keep: set[str] = {
        it.name for it in prog.items
        if isinstance(it, A.FnDecl) and "kernel" in it.attrs
    }
    queue = list(keep)
    while queue:
        fn = fn_by_name.get(queue.pop())
        if fn is None:
            continue
        for callee in _called_fn_names(fn.body):
            if callee in fn_by_name and callee not in keep:
                keep.add(callee)
                queue.append(callee)
    return A.Program(
        module=prog.module,
        items=[
            it for it in prog.items
            if not isinstance(it, A.FnDecl) or it.name in keep
        ],
    )


def _ty_mentions_diff(ty: object) -> bool:
    if ty is None:
        return False
    if isinstance(ty, A.TyGeneric) and ty.base == "D":
        return True
    if dataclasses.is_dataclass(ty):
        for field in dataclasses.fields(ty):
            if _ty_mentions_diff(getattr(ty, field.name)):
                return True
    if isinstance(ty, (list, tuple)):
        return any(_ty_mentions_diff(item) for item in ty)
    return False


def _fn_mentions_diff_signature(fn: A.FnDecl) -> bool:
    if _ty_mentions_diff(fn.return_ty):
        return True
    return any(_ty_mentions_diff(p.ty) for p in fn.params)


def _reachable_function_names(prog: A.Program) -> set[str]:
    fn_by_name = {
        it.name: it for it in prog.items if isinstance(it, A.FnDecl)
    }
    keep: set[str] = {
        it.name for it in prog.items
        if isinstance(it, A.FnDecl)
        and (it.name == "main" or "kernel" in it.attrs)
    }
    queue = list(keep)
    while queue:
        fn = fn_by_name.get(queue.pop())
        if fn is None:
            continue
        for callee in _called_fn_names(fn.body):
            if callee in fn_by_name and callee not in keep:
                keep.add(callee)
                queue.append(callee)
    return keep


def _drop_unreachable_diff_signature_fns(prog: A.Program) -> A.Program:
    """Drop dead D<T>-signature helpers before host/PTX lowering.

    Source-level AD helpers can be valid, typechecked declarations but not
    directly lowerable to host/Tile IR. If they are dead relative to main and
    kernels, they must not prevent strict checks or binary emission for the
    actually emitted program.
    """
    reachable = _reachable_function_names(prog)
    return A.Program(
        module=prog.module,
        items=[
            it for it in prog.items
            if (not isinstance(it, A.FnDecl)
                or not _fn_mentions_diff_signature(it)
                or it.name in reachable)
        ],
    )


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
    "--stdlib", "--no-stdlib", "--hash", "--hash-cons", "--strict",
    "--check-only", "--emit-ast", "--emit-ir", "--emit-asm",
    "--emit-ptx", "--emit-proof-obligations", "--doc", "--help",
    "--no-color", "--color", "-h",
    # Restart 48 B1: --no-opt is documented in HELIX_REFERENCE.md and
    # QUICKSTART.md as a synonym for -O0 and is accepted by both backends
    # (since restart 46). The check.py side was missed in the restart 47
    # B4 flag-parity sweep — that pass only mirrored check.py-only flags
    # into the backends, not the reverse. Treated here as -O0.
    "--no-opt",
})

_KNOWN_WARNING_NAMES = frozenset({"ad", "deprecated"})


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
        elif tok == "--no-opt":
            # Restart 48 B1: synonym for -O0 (matches backend behavior).
            a.opt_level = 0
            a.flags.add(tok)
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
            elif argv[i + 1].startswith("-"):
                errors.append(f"-o requires an output path, got flag: {argv[i + 1]}")
                i += 1
            else:
                a.output = argv[i + 1]
                i += 2
        elif tok == "-l":
            if i + 1 >= n:
                errors.append("-l requires an argument")
                i += 1
            elif argv[i + 1].startswith("-"):
                errors.append(f"-l requires a library name, got flag: {argv[i + 1]}")
                i += 1
            else:
                a.libs.append(argv[i + 1])
                i += 2
        elif tok.startswith("-l") and len(tok) > 2:
            lib_name = tok[2:]
            if lib_name.startswith("-"):
                errors.append(f"-l requires a library name, got flag: {lib_name}")
            else:
                a.libs.append(lib_name)
            i += 1
        elif tok.startswith("-W"):
            body = tok[2:]
            if "=" in body:
                name, val = body.split("=", 1)
                if name not in _KNOWN_WARNING_NAMES:
                    errors.append(f"unknown warning name: {name}")
                    i += 1
                    continue
                if val not in ("warn", "error"):
                    errors.append(
                        f"unknown warning policy for -W{name}: {val}"
                    )
                    i += 1
                    continue
                a.warnings[name] = val
            else:
                if body not in _KNOWN_WARNING_NAMES:
                    errors.append(f"unknown warning name: {body}")
                    i += 1
                    continue
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
    if "--stdlib" in a.flags and "--no-stdlib" in a.flags:
        errors.append("conflicting stdlib flags: choose --stdlib or --no-stdlib")
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
def _drain_ad_warnings_to_records(
    a: "CliArgs",
) -> tuple[list[dict[str, object]], int]:
    """Drain the AD-warning channel and emit diagnostics.

    Returns structured records plus rc=1 if `-Wad=error` was set AND any
    warnings were drained, else rc=0. Must be called on every code path that
    exits successfully from `main()` so that B13 widening warnings emitted
    during typecheck are surfaced even when the user runs with no `--emit-*` /
    `-o` / or `--check-only`.

    The drain itself ALWAYS clears `_DIFF_WARNINGS` (via `take_diff_
    warnings`) — even when the policy is "warn" rather than "error",
    so a subsequent compile in the same process starts clean.
    """
    from .frontend.autodiff import take_diff_warnings
    ad_warnings = take_diff_warnings()
    if not ad_warnings:
        return [], 0
    ad_policy = a.warnings.get("ad", "warn")
    label = "ERROR" if ad_policy == "error" else "warning"
    artifact_stdout = (
        "--emit-proof-obligations" in a.flags
        or "--emit-ptx" in a.flags
        or "--emit-ir" in a.flags
        or "--emit-asm" in a.flags
        or "--emit-ast" in a.flags
    )
    warning_error_mode = any(policy == "error" for policy in a.warnings.values())
    stream = (
        sys.stderr
        if artifact_stdout or ad_policy == "error" or warning_error_mode
        else sys.stdout
    )
    print(f"   ad:        {len(ad_warnings)} {label}(s)", file=stream)
    records = []
    for w in ad_warnings:
        print(f"     helixc: {w}", file=sys.stderr)
        records.append({
            "kind": "ad",
            "policy": ad_policy,
            "message": w,
            "promoted_to_error": ad_policy == "error",
        })
    if ad_policy == "error":
        return records, 1
    return records, 0


def _drain_ad_warnings(a: "CliArgs") -> int:
    """Drain AD warnings for the normal CLI wrapper."""
    _, rc = _drain_ad_warnings_to_records(a)
    return rc


def _abort_if_ad_error_before_artifact(a: "CliArgs") -> int:
    """Promote `-Wad=error` before printing clean markers or artifacts."""
    if a.warnings.get("ad", "warn") != "error":
        return 0
    return _drain_ad_warnings(a)


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


def _atomic_write_bytes(path: str, data: bytes, mode: int | None = None) -> None:
    # Restart 46 B4: catch BaseException (not just OSError) so a
    # KeyboardInterrupt, MemoryError, or any other interruption mid-write
    # still removes the temp file. Previously the broader interruption
    # left a `.<base>.<rand>.tmp` file in the output directory.
    directory = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{base}.",
            suffix=".tmp",
            dir=directory,
        )
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _remove_stale_output(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _same_filesystem_path(left: str, right: str) -> bool:
    left_path = os.path.normcase(os.path.realpath(os.path.abspath(left)))
    right_path = os.path.normcase(os.path.realpath(os.path.abspath(right)))
    return left_path == right_path


def _report_x86_codegen_exception(e: Exception) -> int:
    msg = str(e)
    if isinstance(e, ValueError) and msg.startswith("module has no function "):
        print(f"helixc: codegen error: {msg}", file=sys.stderr)
    else:
        print(
            f"helixc: internal error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        print(
            "helixc: this is a compiler bug - please file an issue.",
            file=sys.stderr,
        )
    return 1


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


def _emit_proof_obligation_artifact(
    path: str | None,
    obligations,
    typecheck_errors,
    pipeline_errors=None,
    input_metadata=None,
    warning_diagnostics=None,
    proof_carries=None,
) -> None:
    pipeline_errors = list(pipeline_errors or [])
    input_metadata = dict(input_metadata or {})
    warning_diagnostics = list(warning_diagnostics or [])
    proof_carries = list(proof_carries or [])
    proof_carry_strategies: dict[str, int] = {}
    for carry in proof_carries:
        data = carry.to_json_dict() if hasattr(carry, "to_json_dict") else dict(carry)
        strategy = data.get("strategy")
        if isinstance(strategy, str):
            proof_carry_strategies[strategy] = (
                proof_carry_strategies.get(strategy, 0) + 1
            )
    artifact = {
        "schema": PROOF_SCHEMA,
        "cache_key": proof_cache_key(input_metadata),
        "path": path,
        "input": input_metadata,
        "summary": {
            "obligations": len(obligations),
            "proof_carries": len(proof_carries),
            "pipeline_errors": len(pipeline_errors),
            "typecheck_errors": len(typecheck_errors),
            "warning_diagnostics": len(warning_diagnostics),
            "warning_errors": sum(
                1 for d in warning_diagnostics
                if d.get("promoted_to_error")
            ),
            "proof_carry_strategies": proof_carry_strategies,
        },
        "obligations": [
            o.to_json_dict() if hasattr(o, "to_json_dict") else dict(o)
            for o in obligations
        ],
        "proof_carries": [
            c.to_json_dict() if hasattr(c, "to_json_dict") else dict(c)
            for c in proof_carries
        ],
        "pipeline_errors": pipeline_errors,
        "typecheck_errors": [str(e) for e in typecheck_errors],
        "warning_diagnostics": warning_diagnostics,
    }
    print(json.dumps(artifact, indent=2, sort_keys=True))


def proof_cache_key(
    input_metadata: dict[str, object],
    *,
    schema: str = PROOF_SCHEMA,
) -> str | None:
    """Return the stable proof-input cache key, or None when no source exists."""
    if input_metadata.get("source_sha256") is None:
        return None
    payload = {
        "schema": schema,
        "input": input_metadata,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _emit_proof_pipeline_error(
    path: str | None,
    phase: str,
    messages: list[str],
    input_metadata=None,
    warning_diagnostics=None,
) -> None:
    if warning_diagnostics is None and input_metadata is not None:
        warning_diagnostics = _proof_input_warning_diagnostics(input_metadata)
    for msg in messages:
        print(msg, file=sys.stderr)
    _emit_proof_obligation_artifact(
        path,
        [],
        [],
        pipeline_errors=[{"phase": phase, "message": "\n".join(messages)}],
        input_metadata=input_metadata,
        warning_diagnostics=warning_diagnostics,
    )


def _proof_input_metadata(
    src_bytes: bytes,
    a: "CliArgs",
    include_stdlib: bool,
) -> dict[str, object]:
    stdlib = _proof_stdlib_manifest(include_stdlib)
    return {
        "source_sha256": hashlib.sha256(src_bytes).hexdigest(),
        "include_stdlib": include_stdlib,
        "stdlib_strict": stdlib["strict"],
        "stdlib_manifest_sha256": stdlib["manifest_sha256"],
        "stdlib_files": stdlib["files"],
        "opt_level": a.opt_level,
        "flags": _proof_normalized_flags(a.flags),
        "libs": list(a.libs),
        "warnings": {k: a.warnings[k] for k in sorted(a.warnings)},
        "color": (
            "auto" if a.color is None
            else "always" if a.color
            else "never"
        ),
    }


def _proof_invocation_input_metadata(a: "CliArgs") -> dict[str, object]:
    include_stdlib = "--no-stdlib" not in a.flags
    if a.path is not None:
        try:
            with open(a.path, "rb") as f:
                return _proof_input_metadata(f.read(), a, include_stdlib)
        except OSError as e:
            source_error = str(e)
    else:
        source_error = "source path is missing"
    stdlib = _proof_stdlib_manifest(include_stdlib)
    return {
        "source_sha256": None,
        "source_available": False,
        "source_error": source_error,
        "include_stdlib": include_stdlib,
        "stdlib_strict": stdlib["strict"],
        "stdlib_manifest_sha256": stdlib["manifest_sha256"],
        "stdlib_files": stdlib["files"],
        "opt_level": a.opt_level,
        "flags": _proof_normalized_flags(a.flags),
        "libs": list(a.libs),
        "warnings": {k: a.warnings[k] for k in sorted(a.warnings)},
        "color": (
            "auto" if a.color is None
            else "always" if a.color
            else "never"
        ),
    }


def _emit_proof_invocation_error(a: "CliArgs", messages: list[str]) -> None:
    _emit_proof_pipeline_error(
        a.path,
        "invocation",
        ["INVOCATION ERROR:"] + [f"  {msg}" for msg in messages],
        input_metadata=_proof_invocation_input_metadata(a),
    )


def _emit_proof_source_read_error(
    a: "CliArgs",
    path: str,
    error: OSError,
) -> None:
    _emit_proof_pipeline_error(
        path,
        "source-read",
        ["SOURCE READ ERROR:", f"  helixc: {error}"],
        input_metadata=_proof_invocation_input_metadata(a),
    )


def _proof_normalized_flags(flags: set[str]) -> list[str]:
    normalized = set(flags)
    # `--stdlib` is an explicit compatibility spelling of the default.
    # Keep `include_stdlib` as the semantic bit and omit this no-op flag so
    # default and explicit-stdlib invocations share the same proof key.
    normalized.discard("--stdlib")
    return sorted(normalized)


def _proof_stdlib_manifest(include_stdlib: bool) -> dict[str, object]:
    files: list[dict[str, object]] = []
    strict = (
        os.environ.get(parser_mod.STDLIB_STRICT_ENV, "").lower()
        in ("1", "true", "yes")
    )
    if include_stdlib:
        stdlib_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(parser_mod.__file__))),
            "stdlib",
        )
        for fname in parser_mod.STDLIB_FILES:
            path = os.path.join(stdlib_dir, fname)
            entry: dict[str, object] = {"path": fname}
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    data = f.read()
                entry["sha256"] = hashlib.sha256(data).hexdigest()
                entry["bytes"] = len(data)
            else:
                entry["missing"] = True
            files.append(entry)
    manifest_src = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "files": files,
        "manifest_sha256": hashlib.sha256(manifest_src).hexdigest(),
        "strict": strict if include_stdlib else False,
    }


def _proof_input_warning_diagnostics(
    input_metadata: dict[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    strict = bool(input_metadata.get("stdlib_strict"))
    for entry in input_metadata.get("stdlib_files", []):
        if not isinstance(entry, dict) or not entry.get("missing"):
            continue
        records.append({
            "kind": "stdlib",
            "policy": "error" if strict else "warn",
            "message": f"stdlib file missing: {entry.get('path')}",
            "promoted_to_error": strict,
            "path": entry.get("path"),
        })
    return records


def _proof_deprecated_warning_diagnostics(
    prog: A.Program,
    a: "CliArgs",
) -> tuple[list[dict[str, object]], int]:
    from .frontend.deprecated_pass import emit_warnings

    policy = a.warnings.get("deprecated", "warn")
    warnings = emit_warnings(prog)
    if not warnings:
        return [], 0
    label = "ERROR" if policy == "error" else "warning"
    print(f"   deprecated: {len(warnings)} {label}(s)", file=sys.stderr)
    records = []
    for w in warnings:
        print(f"     {w}", file=sys.stderr)
        records.append({
            "kind": "deprecated",
            "policy": policy,
            "message": w,
            "promoted_to_error": policy == "error",
        })
    return records, 1 if policy == "error" else 0


def _proof_hard_validation_pipeline_errors(
    prog: A.Program,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def record(phase: str, header: str, diags: list[object]) -> None:
        if not diags:
            return
        lines = [header] + [f"     {d}" for d in diags]
        print(header, file=sys.stderr)
        for d in diags:
            print(f"     {d}", file=sys.stderr)
        errors.append({"phase": phase, "message": "\n".join(lines)})

    from .frontend.trace_pass import validate_trace_attrs
    trace_diags = validate_trace_attrs(prog)
    record("trace", f"   trace:     {len(trace_diags)} ERROR(s)", trace_diags)

    from .frontend.panic_pass import (
        validate_panic_args, validate_unwind,
    )
    panic_diags = validate_panic_args(prog)
    unwind_diags = validate_unwind(prog)
    record("panic", f"   panic:     {len(panic_diags)} ERROR(s)", panic_diags)
    record("unwind", f"   unwind:    {len(unwind_diags)} ERROR(s)", unwind_diags)

    from .frontend.unsafe_pass import check_unsafe_ops
    unsafe_diags = check_unsafe_ops(prog)
    record("unsafe", f"   unsafe:    {len(unsafe_diags)} ERROR(s)", unsafe_diags)

    from .frontend.autotune import validate_autotune_prog
    autotune_diags = validate_autotune_prog(prog)
    record(
        "autotune",
        f"   autotune:  {len(autotune_diags)} ERROR(s)",
        autotune_diags,
    )
    return errors


def _proof_totality_warning_diagnostics(
    prog: A.Program,
    a: "CliArgs",
) -> tuple[list[dict[str, object]], int]:
    fails = check_totality(prog)
    if not fails:
        print(f"   totality:  OK", file=sys.stderr)
        return [], 0
    print(
        f"warning: totality: {len(fails)} fn(s) NOT proven total",
        file=sys.stderr,
    )
    strict = "--strict" in a.flags
    records = []
    for name, reason in fails:
        message = f"totality: {name}: {reason} (trap 21001)"
        print(f"warning: [trap 21001] {message}", file=sys.stderr)
        records.append({
            "kind": "totality",
            "policy": "error" if strict else "warn",
            "message": message,
            "function": name,
            "promoted_to_error": strict,
        })
    if strict:
        print(
            f"\n{len(fails)} totality warning(s); --strict aborts.",
            file=sys.stderr,
        )
        return records, 1
    return records, 0


def _proof_strict_effect_warning_diagnostics(
    prog: A.Program,
    a: "CliArgs",
    include_stdlib: bool,
) -> tuple[list[dict[str, object]], int, list[dict[str, str]]]:
    if "--strict" not in a.flags:
        return [], 0, []

    from .ir.lower_ast import lower
    from .ir.passes.fdce import fdce_module, diagnostic_function_names
    from .ir.passes.const_fold import fold_module, FoldError
    from .ir.passes.cse import cse_module
    from .ir.passes.dce import dce_module
    from .ir.passes.effect_check import (
        check_module as effect_check_module,
        report_diagnostics as report_effect_diagnostics,
        classify_effect_error,
    )
    from .frontend.grad_pass import grad_pass

    try:
        strict_prog = _drop_unreachable_diff_signature_fns(prog)
        grad_pass(strict_prog)
        mod = lower(strict_prog)
    except Exception as e:
        msg = (
            f"strict-effect-check: ERROR\n"
            f"     {type(e).__name__}: {e}"
        )
        print(msg, file=sys.stderr)
        return [], 1, [{"phase": "strict-effect-check", "message": msg}]
    try:
        pre_opt_effect_scope = None
        if include_stdlib:
            pre_opt_effect_scope = diagnostic_function_names(mod)
        pre_opt_eff_errs = effect_check_module(
            mod, only_functions=pre_opt_effect_scope)
        if a.opt_level >= 1:
            fold_module(mod)
            cse_module(mod)
            dce_module(mod)
            fdce_module(mod)
    except FoldError as fe:
        msg = f"helixc: const-fold error: {fe}"
        print(msg, file=sys.stderr)
        return [], 1, [{"phase": "const-fold", "message": msg}]
    except Exception as e:
        msg = (
            f"strict-effect-check: ERROR\n"
            f"     {type(e).__name__}: {e}"
        )
        print(msg, file=sys.stderr)
        return [], 1, [{"phase": "strict-effect-check", "message": msg}]

    try:
        effect_scope = None
        if include_stdlib:
            effect_scope = diagnostic_function_names(mod)
        post_opt_eff_errs = effect_check_module(
            mod, only_functions=effect_scope)
        eff_errs = list(pre_opt_eff_errs)
        seen_eff_errs = set(eff_errs)
        for err in post_opt_eff_errs:
            if err not in seen_eff_errs:
                eff_errs.append(err)
                seen_eff_errs.add(err)

        hard_count = report_effect_diagnostics(eff_errs, stderr=sys.stderr)
        records = []
        for err in eff_errs:
            severity = classify_effect_error(err)
            promoted = severity in ("hard", "unknown")
            records.append({
                "kind": "effect-check",
                "policy": "error" if promoted else "info",
                "severity": severity,
                "message": err,
                "promoted_to_error": promoted,
            })
        if hard_count > 0:
            print(
                f"\n{hard_count} effect-check warning(s); --strict aborts.",
                file=sys.stderr,
            )
            return records, 1, []
        return records, 0, []
    except Exception as e:
        msg = (
            f"strict-effect-check: ERROR\n"
            f"     {type(e).__name__}: {e}"
        )
        print(msg, file=sys.stderr)
        return [], 1, [{"phase": "strict-effect-check", "message": msg}]


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
    proof_mode = "--emit-proof-obligations" in a.flags
    if "--help" in a.flags:
        _print_help()
        return 0

    # Restart 46 B1: clear any stale prior binary at the requested -o path
    # before exiting a bad-invocation path. Without this, a previous
    # successful compile leaves a binary at the target path while the
    # current bad invocation reports an error — callers (CI, tests, users)
    # can mistake the leftover artifact for a successful build of the
    # current invocation.
    #
    # Only safe when the output path is set AND does not match the source
    # path (so we never delete the user's source file). The source==output
    # mismatch is itself caught and reported at a later return path.
    def _cleanup_bad_invocation_output() -> None:
        out = getattr(a, "output", None)
        if out is None:
            return
        src_path = getattr(a, "path", None)
        if src_path is not None and _same_filesystem_path(src_path, out):
            return
        try:
            _remove_stale_output(out)
        except OSError:
            # Best-effort cleanup. The bad-invocation diagnostic is the
            # primary signal; failure to clear stale output here must not
            # mask it. The subsequent compile that would actually need the
            # path clean will re-attempt cleanup with proper error
            # reporting (see the artifact_output_requested block).
            pass

    stdout_modes = {
        "--emit-ast", "--emit-ir", "--emit-asm", "--emit-ptx",
        "--emit-proof-obligations", "--doc",
    }
    selected_stdout_modes = sorted(a.flags & stdout_modes)
    if len(selected_stdout_modes) > 1:
        messages = [
            f"helixc: stdout mode selected: {mode}"
            for mode in selected_stdout_modes
        ]
        messages.append(
            "helixc: choose exactly one stdout-producing mode per invocation"
        )
        if proof_mode:
            _emit_proof_invocation_error(a, messages)
        else:
            for msg in messages:
                print(msg, file=sys.stderr)
        _cleanup_bad_invocation_output()
        return 2
    if errs:
        if proof_mode:
            _emit_proof_invocation_error(
                a,
                [f"helixc: {e}" for e in errs],
            )
        else:
            for e in errs:
                print(f"helixc: {e}", file=sys.stderr)
        _cleanup_bad_invocation_output()
        return 2
    if a.output is not None and selected_stdout_modes:
        messages = [
            f"helixc: {selected_stdout_modes[0]} writes to stdout and cannot be combined with -o"
        ]
        if proof_mode:
            _emit_proof_invocation_error(a, messages)
        else:
            for msg in messages:
                print(msg, file=sys.stderr)
        _cleanup_bad_invocation_output()
        return 2
    if "--check-only" in a.flags and (selected_stdout_modes or a.output is not None):
        target = selected_stdout_modes[0] if selected_stdout_modes else "-o"
        messages = [
            f"helixc: --check-only cannot be combined with {target}"
        ]
        if proof_mode:
            _emit_proof_invocation_error(a, messages)
        else:
            for msg in messages:
                print(msg, file=sys.stderr)
        _cleanup_bad_invocation_output()
        return 2
    artifact_output_requested = (
        a.output is not None
        and not selected_stdout_modes
        and "--check-only" not in a.flags
    )
    if a.path is None:
        if artifact_output_requested:
            try:
                _remove_stale_output(a.output)
            except OSError as e:
                print(
                    f"helixc: cannot clear stale output {a.output!r}: {e}",
                    file=sys.stderr,
                )
                return 1
        if proof_mode:
            _emit_proof_invocation_error(a, ["helixc: source path required"])
            return 2
        if selected_stdout_modes or argv:
            print("helixc: source path required", file=sys.stderr)
            return 2
        _print_help()
        return 2
    path = a.path
    if a.output is not None and _same_filesystem_path(path, a.output):
        print(
            "helixc: output path must differ from input source path",
            file=sys.stderr,
        )
        return 2
    if artifact_output_requested:
        try:
            _remove_stale_output(a.output)
        except OSError as e:
            print(
                f"helixc: cannot clear stale output {a.output!r}: {e}",
                file=sys.stderr,
            )
            return 1
    if not os.path.exists(path):
        if proof_mode:
            _emit_proof_invocation_error(
                a,
                [f"helixc: file not found: {path}"],
            )
            return 2
        print(f"helixc: file not found: {path}", file=sys.stderr)
        return 2

    try:
        with open(path, "rb") as f:
            src_bytes = f.read()
    except OSError as e:
        if proof_mode:
            _emit_proof_source_read_error(a, path, e)
            return 2
        raise
    try:
        src = src_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        if proof_mode:
            proof_input = _proof_input_metadata(
                src_bytes,
                a,
                "--no-stdlib" not in a.flags,
            )
            _emit_proof_pipeline_error(
                path,
                "decode",
                ["DECODE ERROR:", f"  encoding error reading source: {e}"],
                input_metadata=proof_input,
            )
            return 2
        raise

    # Doc extraction is a separate mode: parse not required.
    if "--doc" in a.flags:
        print(extract_doc_comments(src))
        return 0

    artifact_stdout_mode = (
        "--emit-ptx" in a.flags
        or "--emit-ir" in a.flags
        or "--emit-asm" in a.flags
        or "--emit-ast" in a.flags
    )
    warning_error_mode = any(policy == "error" for policy in a.warnings.values())

    def info(msg: str) -> None:
        print(
            msg,
            file=sys.stderr if proof_mode or artifact_stdout_mode or warning_error_mode else sys.stdout,
        )

    diagnostic_stream = sys.stderr if artifact_stdout_mode or warning_error_mode else sys.stdout

    def diag_out(msg: str = "") -> None:
        print(msg, file=diagnostic_stream)

    print(
        f"-- helixc-check: {path}",
        file=sys.stderr if proof_mode or artifact_stdout_mode or warning_error_mode else sys.stdout,
    )
    # Audit 28.8 cycle 2 C2-1: register CliArgs so the outer wrapper
    # can drain AD warnings on ANY return below (including error paths).
    a_holder.append(a)

    include_stdlib = "--no-stdlib" not in a.flags
    proof_input = _proof_input_metadata(src_bytes, a, include_stdlib)

    # 1. Parse
    try:
        prog = parse(src, include_stdlib=include_stdlib)
    except FileNotFoundError as e:
        if proof_mode:
            stdlib_missing = [
                str(d["message"]) for d in
                _proof_input_warning_diagnostics(proof_input)
                if d.get("kind") == "stdlib"
            ]
            _emit_proof_pipeline_error(
                path,
                "stdlib",
                ["STDLIB ERROR:"]
                + [f"  {msg}" for msg in stdlib_missing]
                + ([] if stdlib_missing else [f"  {e}"]),
                input_metadata=proof_input,
            )
            return 2
        raise
    except LexError as e:
        if proof_mode:
            _emit_proof_pipeline_error(
                path, "lex", ["LEX ERROR:", f"  {path}:{e}"],
                input_metadata=proof_input,
            )
            return 1
        print("LEX ERROR:", file=sys.stderr)
        print(f"  {path}:{e}", file=sys.stderr)
        return 1
    except ParseError as e:
        rendered = e.render(
            source=src,
            filename=path,
            color=False if proof_mode else a.color,
        )
        if proof_mode:
            _emit_proof_pipeline_error(
                path,
                "parse",
                ["PARSE ERROR:"] + [f"  {line}"
                                    for line in rendered.splitlines()],
                input_metadata=proof_input,
            )
            return 1
        print("PARSE ERROR:", file=sys.stderr)
        for line in rendered.splitlines():
            print(f"  {line}", file=sys.stderr)
        return 1
    fn_count = sum(1 for it in prog.items if isinstance(it, A.FnDecl))
    info(f"   parse:    OK  ({fn_count} fns, {len(prog.items)} items)")

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

    # 2. Surface-shape rewrites before typecheck. This mirrors the backend
    # driver, so typecheck sees module-local aliases/functions and lifted
    # impl methods rather than silently skipping nested items.
    #
    # Stage 28.9 cycle 66 fix-sweep — flatten modules BEFORE
    # flatten_impls, matching `helixc/backend/x86_64.py` order
    # (lines 3104+3107). Cycle-63 audit found the surface tool
    # skipped flatten_modules entirely so mod-nested @deprecated
    # calls were silently invisible (CN-1 conf 92). Cycle-64
    # added flatten_modules AFTER flatten_impls — wrong order:
    # flatten_impls iterates only top-level items, so it skipped
    # ImplBlocks nested inside ModBlocks; the cycle-65 type-design
    # audit caught this at conf 88. Cycle-66 final fix: align with
    # backend (modules first, then impls).
    #
    # The cycle-66 flatten_modules upgrade rewrites intra-mod calls
    # (e.g. `mod m { fn foo() { foo() } }` becomes `m__foo()
    # { m__foo() }`) so name-based downstream passes (totality
    # self-call detection, deprecated call-site walker) see the
    # mangled names. Pre-upgrade the intra-mod self-call survived
    # un-renamed and totality/deprecated silently missed it.
    from .frontend.flatten_modules import flatten_modules, FlattenError
    try:
        flatten_modules(prog)
    except FlattenError as e:
        if proof_mode:
            _emit_proof_pipeline_error(
                path, "mod-flatten",
                ["   mod-flatten: ERROR", f"     {e}"],
                input_metadata=proof_input,
            )
            return 1
        print(f"   mod-flatten: ERROR", file=sys.stderr)
        print(f"     {e}", file=sys.stderr)
        return 1

    # Audit 28.8 cycle 2 B:C7 — flatten impls so trap 74002
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
        if proof_mode:
            _emit_proof_pipeline_error(
                path, "impl-flatten",
                ["   impl-flatten: ERROR", f"     {e}"],
                input_metadata=proof_input,
            )
            return 1
        diag_out(f"   impl-flatten: ERROR")
        diag_out(f"     {e}")
        return 1

    # Stage 28 — parametric-struct monomorphization (Audit 28.8
    # A3/B1/C1-M2). Surfaces arity-mismatch diagnostics + appends the
    # mono'd StructDecls so downstream passes (lowering, codegen) can
    # find them by mangled name. Doesn't replace typecheck's lookup
    # (which goes through `_resolve_type` -> `mangle_struct` and
    # produces `TyStruct(mangled)` directly).
    from .frontend.struct_mono import monomorphize_structs
    prog, sm_diags = monomorphize_structs(prog)
    if sm_diags:
        if proof_mode:
            _emit_proof_pipeline_error(
                path,
                "struct-mono",
                [f"   struct-mono: {len(sm_diags)} ERROR(s)"]
                + [f"     {d}" for d in sm_diags],
                input_metadata=proof_input,
            )
            return 1
        diag_out(f"   struct-mono: {len(sm_diags)} ERROR(s)")
        for d in sm_diags:
            diag_out(f"     {d}")
        return 1

    # Function monomorphization must run before typecheck for the developer
    # CLI too. The x86 backend already does this; without it, --emit-ir can
    # reject valid generic calls using type aliases while the backend accepts
    # the same source.
    from .frontend.monomorphize import monomorphize_safe
    mono_count, mono_diags = monomorphize_safe(prog)
    if mono_diags:
        if proof_mode:
            _emit_proof_pipeline_error(
                path,
                "fn-mono",
                [f"   fn-mono: {len(mono_diags)} ERROR(s)"]
                + [f"     {d}" for d in mono_diags],
                input_metadata=proof_input,
            )
            return 1
        diag_out(f"   fn-mono: {len(mono_diags)} ERROR(s)")
        for d in mono_diags:
            diag_out(f"     {d}")
        return 1
    if mono_count > 0:
        info(f"   fn-mono: {mono_count} generic instantiation(s)")

    # 2.5 Typecheck after flatten/mono, matching the backend's user-error
    # gate and catching module-local refined aliases before IR emission.
    proof_obligations = []
    proof_carries = []
    if proof_mode:
        tc_errs, proof_obligations, proof_carries = (
            typecheck_with_proof_artifacts(prog)
        )
    else:
        tc_errs = typecheck(prog)
    if tc_errs:
        if proof_mode:
            warning_diagnostics = _proof_input_warning_diagnostics(proof_input)
            ad_diagnostics, _warning_rc = _drain_ad_warnings_to_records(a)
            warning_diagnostics.extend(ad_diagnostics)
            _emit_proof_obligation_artifact(
                path, proof_obligations, tc_errs,
                input_metadata=proof_input,
                warning_diagnostics=warning_diagnostics,
                proof_carries=proof_carries,
            )
            return 1
        diag_out(f"   typecheck: {len(tc_errs)} ERRORS")
        for e in tc_errs[:20]:
            rendered = e.render(source=src, filename=path, color=a.color) \
                if hasattr(e, "render") else str(e)
            for line in rendered.splitlines():
                diag_out(f"     {line}")
        if len(tc_errs) > 20:
            diag_out(f"     ... and {len(tc_errs) - 20} more")
        return 1
    info(f"   typecheck: OK")
    if proof_mode:
        warning_diagnostics = _proof_input_warning_diagnostics(proof_input)
        warning_rc = 1 if any(
            d.get("promoted_to_error") for d in warning_diagnostics
        ) else 0
        proof_pipeline_errors: list[dict[str, str]] = []
        totality_records, totality_rc = _proof_totality_warning_diagnostics(
            prog, a)
        warning_diagnostics.extend(totality_records)
        warning_rc = max(warning_rc, totality_rc)
        deprecated_records, deprecated_rc = _proof_deprecated_warning_diagnostics(
            prog, a)
        warning_diagnostics.extend(deprecated_records)
        warning_rc = max(warning_rc, deprecated_rc)
        proof_pipeline_errors.extend(_proof_hard_validation_pipeline_errors(prog))
        if proof_pipeline_errors:
            warning_rc = max(warning_rc, 1)
        effect_records, effect_rc, effect_pipeline_errors = (
            _proof_strict_effect_warning_diagnostics(
                prog, a, include_stdlib)
        )
        warning_diagnostics.extend(effect_records)
        warning_rc = max(warning_rc, effect_rc)
        proof_pipeline_errors.extend(effect_pipeline_errors)
        ad_diagnostics, ad_rc = _drain_ad_warnings_to_records(a)
        warning_diagnostics.extend(ad_diagnostics)
        warning_rc = max(warning_rc, ad_rc)
        _emit_proof_obligation_artifact(
            path, proof_obligations, tc_errs,
            pipeline_errors=proof_pipeline_errors,
            input_metadata=proof_input,
            warning_diagnostics=warning_diagnostics,
            proof_carries=proof_carries,
        )
        if proof_pipeline_errors:
            return 1
        return warning_rc

    # 3. Totality
    # Stage 28.9 cycle 28 audit-R C27-6 fix (conf 70): pre-fix the
    # diagnostic format diverged from effect-check (and from
    # typecheck's `warning:` convention). Pre-fix output went to
    # STDOUT without a `warning:` prefix; effect-check went to STDERR
    # with `warning:`. A CI runner capturing only stderr missed
    # totality entirely. Now totality follows the same shape as
    # effect-check: per-fail `warning: totality: <fn>: <reason>` to
    # stderr; summary header to stderr; --strict still aborts.
    # The legacy `   totality:  OK` line stays on stdout as a
    # progress indicator for clean compiles (matches `   parse: OK`).
    fails = check_totality(prog)
    if fails:
        # Stage 28.9 cycle 30 audit-R C29-R4 fix (conf 82): the
        # cycle-28 C27-6 migration moved per-fail lines to stderr but
        # left the count summary on stdout — partial migration. Now
        # the count also goes to stderr with `warning:` prefix so a
        # CI runner capturing stderr sees the full picture.
        print(
            f"warning: totality: {len(fails)} fn(s) NOT proven total",
            file=sys.stderr,
        )
        for name, reason in fails:
            print(
                f"warning: [trap 21001] totality: {name}: {reason}",
                file=sys.stderr,
            )
        if "--strict" in a.flags:
            print(
                f"\n{len(fails)} totality warning(s); --strict aborts.",
                file=sys.stderr,
            )
            return 1
    else:
        info(f"   totality:  OK")

    # Stage 28.7: @deprecated pass. Runs the Python-side walker and
    # collects warnings; the -Wdeprecated=error flag promotes them.
    from .frontend.deprecated_pass import emit_warnings
    deprecate_policy = a.warnings.get("deprecated", "warn")
    warnings = emit_warnings(prog)
    if warnings:
        label = "ERROR" if deprecate_policy == "error" else "warning"
        diag_out(f"   deprecated: {len(warnings)} {label}(s)")
        for w in warnings:
            diag_out(f"     {w}")
        if deprecate_policy == "error":
            return 1

    # Stage 25: @trace validation pass (Audit 28.8 A7).
    # `validate_trace_attrs` rejects @trace on extern "C" fns (no body
    # to instrument). Codegen-side wiring (IR ops TRACE_ENTRY /
    # TRACE_EXIT) is handled in ir/lower_ast.py + backend/x86_64.py.
    from .frontend.trace_pass import validate_trace_attrs
    trace_diags = validate_trace_attrs(prog)
    if trace_diags:
        diag_out(f"   trace:     {len(trace_diags)} ERROR(s)")
        for d in trace_diags:
            diag_out(f"     {d}")
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
        diag_out(f"   panic:     {len(panic_diags)} ERROR(s)")
        for d in panic_diags:
            diag_out(f"     {d}")
    if unwind_diags:
        diag_out(f"   unwind:    {len(unwind_diags)} ERROR(s)")
        for d in unwind_diags:
            diag_out(f"     {d}")
    if panic_diags or unwind_diags:
        return 1

    # Stage 28.6: unsafe capability gate (Audit 28.8 A2).
    # `check_unsafe_ops` flags raw-pointer ops (Unary deref) outside
    # any enclosing UnsafeBlock with trap 28601. Empty list = clean.
    from .frontend.unsafe_pass import check_unsafe_ops
    unsafe_diags = check_unsafe_ops(prog)
    if unsafe_diags:
        diag_out(f"   unsafe:    {len(unsafe_diags)} ERROR(s)")
        for d in unsafe_diags:
            diag_out(f"     {d}")
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
        diag_out(f"   autotune:  {len(autotune_diags)} ERROR(s)")
        for d in autotune_diags:
            diag_out(f"     {d}")
        return 1


    # 4. Optional hash dump
    if "--hash" in a.flags:
        diag_out("   hashes:")
        for it in prog.items:
            if isinstance(it, A.FnDecl):
                diag_out(
                    f"     {it.name:<40} {short_hash(structural_hash(it))}"
                )

    # 4.5 Optional hash-cons
    if "--hash-cons" in a.flags:
        n_shared = hash_cons(prog)
        diag_out(f"   hash-cons: {n_shared} AST node(s) deduped")

    # --check-only short-circuit: stop here.
    # Stage 28.9 cycle 26 audit-R C25-1 NOTE (conf 92): the audit
    # flagged that effect_check (added in cycle 24) is gated on the
    # emit-* / -o block below, so `--check-only` and default-no-flag
    # invocations bypass IR-level effect verification. This gating
    # is INTENTIONAL — `--check-only` is documented as "fast lexical
    # / type / totality validation" (see existing test
    # test_ad_drain_runs_on_check_only); running lower+fold+effect_check
    # in this path would 2x-10x the cost depending on program size.
    # Users who want full effect verification should use `--emit-ir`
    # (lowers + folds + effect-checks without writing an ELF).
    # The IR-level effect_check is reachable from any emit path
    # (--emit-ir / --emit-asm / --emit-ptx / -o) AND from x86_64.py's
    # backend driver — so any compile that produces a binary goes
    # through it. Cycle 26 documents the trade-off explicitly.
    if "--check-only" in a.flags:
        ad_rc = _abort_if_ad_error_before_artifact(a)
        if ad_rc != 0:
            return ad_rc
        diag_out("-- clean (check-only)")
        return 0

    # 5. Lower + (optional) optimization passes
    if any(f in a.flags for f in ("--emit-ir", "--emit-asm", "--emit-ptx")) \
            or "--strict" in a.flags \
            or a.output is not None:
        from .ir.lower_ast import lower
        from .ir.passes.fdce import fdce_module, diagnostic_function_names
        from .ir.passes.const_fold import fold_module, FoldError
        from .ir.passes.cse import cse_module
        from .ir.passes.dce import dce_module
        # Stage 28.9 cycle 24 audit-R C23-3 (conf 85): import effect_check
        # so the optimization pipeline can mirror x86_64.py's authoritative
        # IR-level effect verification. Pre-fix, check.py never ran
        # effect_check at all — a @pure violation that survived fold/cse/dce
        # silently emitted IR/ELF via check.py while x86_64.py would refuse.
        # Stage 28.9 cycle 30 audit-R C29-R2 (conf 88): co-locate
        # `classify_effect_error` with `check_module` so the lazy-import
        # cluster is consistent — cycle-28 left it as a second inline
        # import deeper in the function body, an asymmetry that could
        # confuse import-cycle audits.
        from .ir.passes.effect_check import (
            check_module as effect_check_module,
            report_diagnostics as report_effect_diagnostics,
        )
        # Audit 28.8 B5: invoke grad_pass before lowering so any
        # `grad(loss)` calls are rewritten to symbolic gradients (and
        # the AD passes' diagnostics accumulate in _DIFF_WARNINGS).
        # We import lazily so check.py's start-up cost stays low for
        # programs that don't use grad().
        from .frontend.grad_pass import grad_pass
        ptx_full_eff_errs = []
        if "--emit-ptx" in a.flags:
            try:
                ptx_full_prog = _drop_unreachable_diff_signature_fns(prog)
                grad_pass(ptx_full_prog)
                ptx_full_mod = lower(ptx_full_prog)
                ptx_full_scope = None
                if include_stdlib:
                    ptx_full_scope = diagnostic_function_names(
                        ptx_full_mod)
                ptx_full_eff_errs = effect_check_module(
                    ptx_full_mod, only_functions=ptx_full_scope)
            except Exception as e:
                print(
                    f"helixc: PTX validation error: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                return 1
        lower_prog = prog
        if "--emit-ptx" in a.flags:
            lower_prog = _kernel_reachable_program(prog)
        else:
            lower_prog = _drop_unreachable_diff_signature_fns(prog)
        grad_pass(lower_prog)
        mod = lower(lower_prog)
        pre_opt_effect_scope = None
        if include_stdlib:
            pre_opt_effect_scope = diagnostic_function_names(mod)
        pre_opt_eff_errs = effect_check_module(
            mod, only_functions=pre_opt_effect_scope)
        # Optimization pipeline (Audit 28.8 A10, Stage 31 parity):
        #   -O0          — no opts
        #   -O1 (default) mirrors x86_64.py default for host IR/ELF:
        #                  const_fold + cse + dce + fdce
        #                  (`--emit-ptx` keeps DCE/FDCE off because the
        #                  textual kernel body is the inspected artifact)
        #   -O2          same pass set for now
        #   -O3          same pass set for now; reserved for a future
        #                  aggressive layer.
        #
        # The pre-fix code ran ONLY fdce at -O1+ and stubbed -O2 with an
        # empty try/import-pass placeholder — so users invoking `-O2`
        # got the SAME IR as `-O1`, despite the help text promising
        # +cse+dce. The fix wires the real passes; the placeholder is
        # removed.
        # Stage 28.9 cycle 24 audit-R C23-1 fix (conf 92): wrap the
        # optimization-pass calls in a FoldError-aware try/except so
        # that a user-authored compile-time NaN (trap 17001) or out-of-
        # range constant shift (trap 17002) renders as a clean
        # `helixc: const-fold error: [trap 1700N] ...` diagnostic with
        # rc=1, instead of bubbling up to the outer Exception handler
        # at main() which prints "this is a compiler bug — please file
        # an issue." (The latter is the WRONG message for a source-
        # level error.) Mirrors the FoldError-as-user-diagnostic
        # pattern used implicitly by x86_64.py.
        try:
            emit_ptx = "--emit-ptx" in a.flags
            if a.opt_level >= 1:
                fold_module(mod)
                cse_module(mod)
                if any(fn.attrs.get("kernel") for fn in mod.functions.values()):
                    from .backend.ptx import validate_kernel_tile_lowering
                    try:
                        validate_kernel_tile_lowering(mod)
                    except Exception as e:
                        print(
                            f"helixc: PTX validation error: {e}",
                            file=sys.stderr,
                        )
                        return 1
                if not emit_ptx:
                    dce_module(mod)
                    fdce_module(mod)
            if a.opt_level >= 2:
                # Same pass set as -O1 until a stronger layer lands.
                pass
            if a.opt_level >= 3:
                # No aggressive layer yet — kept distinct from -O2 so a
                # future pass can wire in without re-shaping the
                # dispatch.
                pass
        except FoldError as fe:
            print(
                f"helixc: const-fold error: {fe}",
                file=sys.stderr,
            )
            return 1
        if a.opt_level == 0 and not emit_ptx \
                and any(fn.attrs.get("kernel") for fn in mod.functions.values()):
            from .backend.ptx import validate_kernel_tile_lowering
            try:
                validate_kernel_tile_lowering(mod)
            except Exception as e:
                print(
                    f"helixc: PTX validation error: {e}",
                    file=sys.stderr,
                )
                return 1

        # Stage 28.9 cycle 24 → cycle 26 → cycle 28 evolution: effect-
        # Strict mode must not fail on unused auto-bundled stdlib fns when
        # -O0 skips the normal FDCE optimization pipeline.
        effect_scope = None
        if include_stdlib:
            effect_scope = diagnostic_function_names(mod)

        # Effect-check classification was previously done by inline substring
        # matching duplicated across this driver and x86_64.py. Cycle 28
        # audit-R C27-2/C27-3 refactored it: the canonical classifier
        # lives in effect_check.classify_effect_error, so the message
        # format is owned by one module and the trap-id discrimination
        # is identical in both drivers.
        #
        # Severity classes (per effect_check.py docstring lines 296-303):
        #   - "hard"    → trap 19001 @pure or under-declared violation.
        #     Warning by default; --strict aborts. Cycle 26 audit-T
        #     C24-1 (conf 95): pre-fix this rejected
        #     helixc/examples/hello_world.hx via check.py while
        #     x86_64.py compiled it cleanly — wrong asymmetry.
        #   - "info"    → trap 19002 declared-unused effect. Never
        #     causes failure ("a code smell, not a correctness
        #     violation" per docstring).
        #   - "unknown" → fail-closed. A new trap-id added in a future
        #     stage falls into this bucket and is treated as hard so
        #     a fresh hardening is never silently downgraded. Cycle 28
        #     audit-R C27-3 (conf 78).
        # Stage 28.9 cycle 30 audit-R C29-5 (conf 68): the per-line
        # dispatch loop is now in `effect_check.report_diagnostics`
        # so check.py and x86_64.py share one printing implementation.
        post_opt_eff_errs = effect_check_module(
            mod, only_functions=effect_scope)
        eff_errs = list(ptx_full_eff_errs)
        for err in pre_opt_eff_errs:
            if err not in eff_errs:
                eff_errs.append(err)
        seen_eff_errs = set(eff_errs)
        for err in post_opt_eff_errs:
            if err not in seen_eff_errs:
                eff_errs.append(err)
                seen_eff_errs.add(err)
        hard_count = report_effect_diagnostics(eff_errs, stderr=sys.stderr)
        if hard_count > 0 and "--strict" in a.flags:
            print(
                f"\n{hard_count} effect-check warning(s); --strict aborts.",
                file=sys.stderr,
            )
            return 1

        # Audit 28.8 cycle 2 C2-1: the AD-warning drain was moved out
        # of this branch into the outer `main()` wrapper. Pre-fix, the
        # drain only ran when an `--emit-*` flag or `-o` was set, so
        # default-no-emit and `--check-only` silently lost the B13
        # widening warnings emitted during typecheck. Drain is now
        # universal — see `_drain_ad_warnings` and the `main` wrapper.

    if "--emit-ir" in a.flags:
        ad_rc = _abort_if_ad_error_before_artifact(a)
        if ad_rc != 0:
            return ad_rc
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
        ad_rc = _drain_ad_warnings(a)
        if ad_rc != 0:
            return ad_rc
        # Audit 28.8 A9: wrap backend call in try/except so internal
        # compiler errors render as `helixc: internal error: ...`
        # instead of leaking Python tracebacks. Mirrors the existing
        # --emit-ptx error-path pattern (line 422-427).
        try:
            elf = compile_module_to_elf(mod)
        except Exception as e:
            return _report_x86_codegen_exception(e)
        # Phase-0: ELF hex dump. A real disassembler would use objdump.
        print(f"   asm: {len(elf)} bytes of ELF (use `objdump -d` for asm)")
        for i in range(0, len(elf), 16):
            row = elf[i:i + 16]
            hex_row = " ".join(f"{b:02x}" for b in row)
            print(f"     {i:08x}  {hex_row}")
        return 0

    if "--emit-ptx" in a.flags:
        from .ir.tile_ir import lower_to_tile
        from .backend.ptx import emit_ptx, kernel_only_module
        try:
            kernel_mod = kernel_only_module(mod)
            tile_mod = lower_to_tile(kernel_mod)
            ptx = emit_ptx(tile_mod)
            ad_rc = _drain_ad_warnings(a)
            if ad_rc != 0:
                return ad_rc
            print(ptx)
        except Exception as e:
            print(f"   ptx: backend error: {e}", file=sys.stderr)
            return 1
        return 0

    # 6. Codegen to ELF (when -o)
    if a.output is not None:
        from .backend.x86_64 import compile_module_to_elf
        ad_rc = _drain_ad_warnings(a)
        if ad_rc != 0:
            return ad_rc
        # Audit 28.8 A9: wrap backend call so internal codegen errors
        # render cleanly. Also catch OSError on the file write so
        # permission / disk-full failures surface as a real diagnostic,
        # not a partial-file + traceback.
        try:
            elf = compile_module_to_elf(mod)
        except Exception as e:
            return _report_x86_codegen_exception(e)
        try:
            _atomic_write_bytes(a.output, elf, mode=0o755)
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

    ad_rc = _abort_if_ad_error_before_artifact(a)
    if ad_rc != 0:
        return ad_rc
    print("-- clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
