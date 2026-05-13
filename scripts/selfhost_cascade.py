"""Run a multi-generation Helix self-host cascade.

The cascade builds a seed compiler with the Python reference path, then uses
that compiler to rebuild the same Helix compiler source repeatedly:

    G1 -> G2 -> G3 -> ... -> G{N+1}

For a stable self-host, G2..G{N+1} must be byte-identical. G1 may differ
because it comes from the seed/reference path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helixc.backend.x86_64 import compile_module_to_elf
from helixc.frontend.flatten_impls import flatten_impls
from helixc.frontend.flatten_modules import flatten_modules
from helixc.frontend.grad_pass import grad_pass
from helixc.frontend.monomorphize import monomorphize
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.cse import cse_module
from helixc.ir.passes.dce import dce_module
from helixc.ir.passes.fdce import fdce_module


def compile_seed_elf(src: str) -> bytes:
    prog = parse(src, include_stdlib=True)
    flatten_modules(prog)
    flatten_impls(prog)
    monomorphize(prog)
    grad_pass(prog)
    mod = lower(prog)
    fold_module(mod)
    cse_module(mod)
    dce_module(mod)
    fdce_module(mod)
    return compile_module_to_elf(mod)


def bootstrap_source(input_path: str, output_path: str) -> str:
    lexer = (ROOT / "helixc" / "bootstrap" / "lexer.hx").read_text()
    lexer_no_main = lexer.rsplit(
        "// --------------------------------------------------------------\n// Demo:",
        1,
    )[0]
    parser_body = (ROOT / "helixc" / "bootstrap" / "parser.hx").read_text()
    kovc = (ROOT / "helixc" / "bootstrap" / "kovc.hx").read_text()
    kovc_lib = kovc.rsplit(
        "// --------------------------------------------------------------\n// Demo:",
        1,
    )[0]
    driver = f"""
fn main() -> i32 {{
    let src_start = __arena_len();
    let src_len = read_file_to_arena("{input_path}");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("{output_path}", elf_start, total)
}}
"""
    return lexer_no_main + parser_body + kovc_lib + driver


def run_wsl(cmd: str, *, data: bytes | None = None,
            timeout: int = 120) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["wsl", "-e", "bash", "-c", cmd],
        input=data,
        capture_output=True,
        timeout=timeout,
    )


def write_wsl(path: str, data: bytes, *, executable: bool = False) -> None:
    qpath = shlex.quote(path)
    proc = run_wsl(f"cat > {qpath}", data=data, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to write {path}: stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    if executable:
        proc = run_wsl(f"chmod +x {qpath}", timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"failed to chmod {path}: stderr={proc.stderr!r}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(path: str) -> tuple[int, str]:
    qpath = shlex.quote(path)
    proc = run_wsl(
        f"stat -c%s {qpath} && sha256sum {qpath} | awk '{{print $1}}'",
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to inspect {path}: stderr={proc.stderr!r}")
    lines = proc.stdout.decode("utf-8").strip().splitlines()
    return int(lines[0]), lines[1]


def build_next(generation: int, compiler_path: str, next_path: str,
               output_path: str) -> dict[str, str | int]:
    qcompiler = shlex.quote(compiler_path)
    qnext = shlex.quote(next_path)
    qout = shlex.quote(output_path)
    cmd = (
        f"rm -f {qout}; "
        f"chmod +x {qcompiler}; "
        f"{qcompiler} >/tmp/helix_cascade_g{generation}.stdout "
        f"2>/tmp/helix_cascade_g{generation}.stderr; "
        f"rc=$?; "
        f"if [ ! -s {qout} ]; then "
        f"  echo __NO_OUTPUT__ rc=$rc; "
        f"  cat /tmp/helix_cascade_g{generation}.stderr; "
        f"  exit 91; "
        f"fi; "
        f"cp {qout} {qnext}; chmod +x {qnext}; "
        f"size=$(stat -c%s {qnext}); "
        f"sha=$(sha256sum {qnext} | awk '{{print $1}}'); "
        f"echo exit_low_byte=$rc size=$size sha=$sha"
    )
    proc = run_wsl(cmd, timeout=180)
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"generation {generation} failed: stdout={text!r} "
            f"stderr={proc.stderr!r}"
        )
    fields: dict[str, str | int] = {"line": text}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = int(value) if key in {"exit_low_byte", "size"} else value
    return fields


def run_smoke(final_compiler: str, input_path: str, output_path: str) -> None:
    cases = [
        ("literal", "fn main() -> i32 { 42 }", 42),
        (
            "call",
            "fn add(a: i32, b: i32) -> i32 { a + b } "
            "fn main() -> i32 { add(20, 22) }",
            42,
        ),
        (
            "loop",
            "fn main() -> i32 { let mut x = 0; "
            "while x < 42 { x = x + 1; } x }",
            42,
        ),
    ]
    for idx, (name, src, expected) in enumerate(cases, 1):
        write_wsl(input_path, src.encode("utf-8"))
        qcompiler = shlex.quote(final_compiler)
        qout = shlex.quote(output_path)
        smoke_bin = shlex.quote(f"/tmp/helix_cascade_smoke_{idx}.bin")
        cmd = (
            f"rm -f {qout} {smoke_bin}; "
            f"{qcompiler} >/tmp/helix_cascade_smoke_{idx}.compiler.stdout "
            f"2>/tmp/helix_cascade_smoke_{idx}.compiler.stderr; "
            f"if [ ! -s {qout} ]; then echo __NO_SMOKE_OUTPUT__; exit 92; fi; "
            f"cp {qout} {smoke_bin}; chmod +x {smoke_bin}; "
            f"{smoke_bin}; echo exit=$?"
        )
        proc = run_wsl(cmd, timeout=60)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0 or f"exit={expected}" not in stdout:
            raise RuntimeError(
                f"smoke case {name!r} failed: stdout={stdout!r} "
                f"stderr={proc.stderr!r}"
            )
        print(f"smoke {name}: exit={expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--prefix", default="/tmp/helix_cascade")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if args.generations < 2:
        raise SystemExit("--generations must be at least 2")

    input_path = f"{args.prefix}_src.hx"
    output_path = f"{args.prefix}_next.bin"

    print(f"cascade: seed + {args.generations} self-host rebuilds")
    source = bootstrap_source(input_path, output_path)
    seed = compile_seed_elf(source)
    seed_path = f"{args.prefix}_g1.bin"
    write_wsl(input_path, source.encode("utf-8"))
    write_wsl(seed_path, seed, executable=True)
    print(f"G1 seed: size={len(seed)} sha={sha256_bytes(seed)}")

    generations: list[dict[str, str | int]] = []
    compiler_path = seed_path
    for generation in range(1, args.generations + 1):
        next_path = f"{args.prefix}_g{generation + 1}.bin"
        info = build_next(generation, compiler_path, next_path, output_path)
        generations.append(info)
        print(
            f"G{generation + 1}: exit_low_byte={info['exit_low_byte']} "
            f"size={info['size']} sha={info['sha']}"
        )
        compiler_path = next_path

    stable_hashes = {str(info["sha"]) for info in generations}
    stable_sizes = {int(info["size"]) for info in generations}
    if len(stable_hashes) != 1 or len(stable_sizes) != 1:
        print("cascade: FAILED")
        print(f"unique hashes: {sorted(stable_hashes)}")
        print(f"unique sizes: {sorted(stable_sizes)}")
        return 2

    print(
        "cascade: PASS "
        f"G2..G{args.generations + 1} are byte-identical "
        f"sha={next(iter(stable_hashes))}"
    )
    run_smoke(compiler_path, input_path, output_path)
    print("smoke: PASS final generation compiled and ran all smoke programs")

    if not args.keep:
        cleanup = (
            f"rm -f {shlex.quote(args.prefix)}_src.hx "
            f"{shlex.quote(args.prefix)}_next.bin "
            f"{shlex.quote(args.prefix)}_g*.bin "
            f"/tmp/helix_cascade_g*.stdout /tmp/helix_cascade_g*.stderr "
            f"/tmp/helix_cascade_smoke_*.bin "
            f"/tmp/helix_cascade_smoke_*.stdout "
            f"/tmp/helix_cascade_smoke_*.stderr"
        )
        run_wsl(cleanup, timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
