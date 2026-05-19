"""
Run one or all of the foundation demo programs.

Usage:
    python -m helixc.examples.run                # run all 5 in sequence
    python -m helixc.examples.run mandelbrot     # just the Mandelbrot
    python -m helixc.examples.run --list         # list available demos

Each demo is a `.hx` source file in this directory. The runner compiles
it through the same pipeline the test suite uses (parse -> grad_pass ->
lower -> opt -> codegen -> ELF), drops the binary into helixc/tests/_tmp,
runs it via WSL, and prints the result alongside the expected outcome.

The Mandelbrot demo prints its visualization directly to stdout; the
other four use exit codes to signal results (Helix's only output channels
in Phase 0 are print_str / print_int / process exit code).

License: Apache 2.0
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import Optional

from ..backend.x86_64 import compile_module_to_elf
from ..frontend.grad_pass import grad_pass
from ..frontend.parser import parse
from ..ir.lower_ast import lower
from ..ir.passes.const_fold import fold_module
from ..ir.passes.cse import cse_module
from ..ir.passes.dce import dce_module
from ..ir.passes.fdce import fdce_module


DEMOS: dict[str, dict] = {
    "mandelbrot": {
        "file": "mandelbrot.hx",
        "title": "Mandelbrot ASCII fractal (60x22 grid, 10 shading levels)",
        "expects": "ASCII fractal rendered to stdout; cardioid + period-2 bulb visible",
    },
    "metacircular": {
        "file": "metacircular_eval.hx",
        "title": "Metacircular evaluator (Helix interpreting Helix's own AST)",
        "expects": "exit code 40  (= eval(let x=5 in if x<10 then x*(x+3) else x-99))",
    },
    "symbolic": {
        "file": "symbolic_algebra.hx",
        "title": "Symbolic algebra engine (differentiate + simplify + evaluate)",
        "expects": "exit code 77  (= d/dx(x^3 + 2x) at x=5 = 3*25 + 2)",
    },
    "sat": {
        "file": "sat_solver.hx",
        "title": "DPLL Boolean SAT solver (4-var 7-clause 3-SAT)",
        "expects": "exit code 1   (= satisfiable)",
    },
    "graddescent": {
        "file": "helix_grad_descent.hx",
        "title": "Helix differentiates Helix (compile-time AD + runtime SGD)",
        "expects": "exit code 43-44  (= w*100 mod 256 after 50 SGD steps; w -> 3)",
    },
    "provenance": {
        "file": "dogfood_06_provenance_datalog.hx",
        "title": "Datalog-shaped reasoning over Logic<i32> (Stage 36 Inc 4)",
        "expects": "exit code 42  (grandparent rule fires + tautology holds for both P)",
    },
    "fuzzysgd": {
        "file": "dogfood_07_provenance_sgd.hx",
        "title": "SGD learns a fuzzy-logic rule via gradients-through-Logic (Stage 36 Inc 7)",
        "expects": "exit code 42  (w converges to 0.8 from grad_rev(fuzzy_and(0.5, w) - 0.4)^2)",
    },
    "twoparam": {
        "file": "dogfood_08_two_param_fuzzy_rule.hx",
        "title": "Two-param SGD learns a fuzzy_or(fuzzy_and(...)) rule (Stage 36 Inc 8)",
        "expects": "exit code 42  (w1→0.9, w2→0.7 via indexed grad_rev across multi-arg loss)",
    },
    "kgraph": {
        "file": "dogfood_09_knowledge_graph.hx",
        "title": "Knowledge-graph reasoner with provenance recovery (Stage 36 Inc 10)",
        "expects": "exit code 42  (3 facts + 2 chained grandparent rules + parent_*_at provenance recovery)",
    },
    "memtiers": {
        "file": "dogfood_10_memory_tiers.hx",
        "title": "Memory-tier lifecycle reasoner (Stage 37 Inc 2)",
        "expects": "exit code 42  (3 observations cycle through working->episodic->semantic->working + procedural sanity)",
    },
    "frames": {
        "file": "dogfood_11_spatial_frames.hx",
        "title": "Spatial-frame lifecycle reasoner (Stage 38 Inc 3)",
        "expects": "exit code 42  (3 observations cycle through WorldFrame->RobotFrame->CameraFrame->WorldFrame)",
    },
    "temporal": {
        "file": "dogfood_12_temporal_lifecycle.hx",
        "title": "Temporal-kind lifecycle reasoner (Stage 39 Inc 3)",
        "expects": "exit code 42  (3 observations cycle through Present->Future->Present->Past + recall_past + Eternal sanity)",
    },
    "modal": {
        "file": "dogfood_13_modal_lifecycle.hx",
        "title": "Modal/epistemic lifecycle reasoner (Stage 40 Inc 3)",
        "expects": "exit code 42  (3 goals achieved + 3 beliefs confirmed + Uncertain sanity + Known<Past<i32>> cross-stage composition)",
    },
    "causal": {
        "file": "dogfood_14_causal_lifecycle.hx",
        "title": "Causal/intent lifecycle reasoner (Stage 41 Inc 3)",
        "expects": "exit code 42  (3 propositions cycle Cause->Effect->Joint->Independent + Known<Cause<i32>> 5-stack composition)",
    },
    "planning": {
        "file": "dogfood_15_agi_planning_loop.hx",
        "title": "AGI quintet cohesion: planning-loop with 4-deep wrapper stack (Stage 42 Inc 1)",
        "expects": "exit code 42  (3 observations cycle Known<Present<WorldFrame<i32>>> -> ... -> Believed<Future<WorldFrame<Effect<i32>>>> -> unwrap; all 4 wrapper layers identity-lower without value drift)",
    },
    "result": {
        "file": "dogfood_16_result_basic.hx",
        "title": "Result<T,E> basic round-trip (Stage 46 Inc 3)",
        "expects": "exit code 42  (3 safe_double via Ok/map_ok/unwrap_ok + Result<Known<i32>, i32> cross-stack composition)",
    },
    "try": {
        "file": "dogfood_17_try_operator.hx",
        "title": "? propagation operator over Result<T,E> (Stage 48 Inc 3)",
        "expects": "exit code 42  (chained safe_div via ? operator: 20/4=5, 5/1=5, 5+37=42; Phase-0 identity-lowered)",
    },
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_and_run(src_path: str, timeout: int = 120) -> tuple[str, str, int]:
    """Compile a Helix source file and run the resulting ELF via WSL.
    Returns (stdout, stderr, exit_code). v2.2 polish item 8 (RT M2):
    stderr was previously discarded; now propagated so runtime panics
    and WSL diagnostics are visible."""
    src = open(src_path).read()
    prog = parse(src, include_stdlib=True)
    grad_pass(prog)
    mod = lower(prog)
    fold_module(mod)
    cse_module(mod)
    dce_module(mod)
    fdce_module(mod)
    elf = compile_module_to_elf(mod)
    proj = _project_root()
    out_dir = os.path.join(proj, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    h = hashlib.sha256(elf).hexdigest()[:12]
    out_path = os.path.join(out_dir, f"demo_{h}.bin")
    # Restart 46 B5: write atomically (temp file + replace + cleanup on
    # failure) so a partial write or interruption never leaves a
    # half-written binary at out_path. Mirrors the canonical
    # _atomic_write_bytes pattern in helixc.check.
    import tempfile
    directory = os.path.dirname(os.path.abspath(out_path)) or "."
    base = os.path.basename(out_path)
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{base}.",
            suffix=".tmp",
            dir=directory,
        )
        with os.fdopen(fd, "wb") as f:
            f.write(elf)
        os.chmod(tmp_path, 0o755)
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    rel = os.path.relpath(out_path, proj).replace("\\", "/")
    wsl_path = f"/mnt/c/Projects/Kovostov-Native/{rel}"
    result = subprocess.run(
        ["wsl", "--", "bash", "-c", f"chmod +x {wsl_path} && {wsl_path}"],
        capture_output=True,
        timeout=timeout,
    )
    # v2.2 polish item 8 (RT M2 from v2.1 5-clean-gate): the WSL
    # subprocess's stderr buffer was previously discarded entirely;
    # if wsl was missing, the binary segfaulted, or runtime panicked,
    # the user saw an empty body / "exit code 139" with zero
    # diagnostic. Stderr contains the real failure — surface it.
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    return stdout, stderr, result.returncode


def _run_one(key: str) -> bool:
    """Run a single demo by short name. Return True on success
    (i.e., compile+run returned exit code 0). v2.2 polish item 8
    (RT M3 from v2.1 5-clean-gate): the prior implementation always
    returned True, so CI smoke tests of `python -m helixc.examples.run`
    would pass even when every demo segfaulted/panicked/build-failed.
    R1 fix: propagate `code == 0` to the caller."""
    if key not in DEMOS:
        print(f"unknown demo {key!r}; use --list to see available demos")
        return False
    info = DEMOS[key]
    proj = _project_root()
    src_path = os.path.join(proj, "helixc", "examples", info["file"])
    bar = "=" * 70
    print(bar)
    print(f"  {key.upper()}: {info['title']}")
    print(f"  source: helixc/examples/{info['file']}")
    print(f"  expect: {info['expects']}")
    print(bar)
    out, err, code = _build_and_run(src_path)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        # v2.2 polish item 8: print stderr to the host stderr so the
        # user sees runtime panics + WSL diagnostics, not just stdout.
        import sys as _sys
        print(err, end="" if err.endswith("\n") else "\n", file=_sys.stderr)
    print(f"  -> exit code {code}")
    print()
    return code == 0


def _list() -> None:
    for key, info in DEMOS.items():
        print(f"  {key:14s} {info['title']}")


def _help() -> None:
    """Restart 61 B4: print usage / help. Pre-fix, the runner had no
    `-h` / `--help` discoverability — users had to read the module
    docstring to learn about `--list` or the per-demo short names."""
    print("Usage:")
    print("    python -m helixc.examples.run                # run all demos")
    print("    python -m helixc.examples.run <demo>...      # run one or more demos")
    print("    python -m helixc.examples.run --list         # list demo short names")
    print("    python -m helixc.examples.run -l             # alias for --list")
    print("    python -m helixc.examples.run -h | --help    # this message")
    print()
    print("Demos:")
    _list()


def main(argv: Optional[list[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if "-h" in args or "--help" in args:
        _help()
        return 0
    if "--list" in args or "-l" in args:
        _list()
        return 0
    if not args:
        # Run all in stable order.
        for key in DEMOS:
            _run_one(key)
        return 0
    for key in args:
        if not _run_one(key):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
