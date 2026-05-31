#!/usr/bin/env python3
"""Equivalence battery: seed-built kovc (K1') vs Python-built kovc (K1py) over a
diverse program corpus -- the REPRODUCIBLE form of the cross-check.

`ddc_check.py` proves the two independently-built kovc compilers agree byte-for-byte
on ONE input (the 1.5 MB self-source). That is the canonical Wheeler fixpoint, but a
single input. This battery extends the equivalence to a committed corpus of small
diverse programs in the Helix self-hosting subset: each program is compiled by BOTH
K1' (seed-built kovc) and K1py (Python-built kovc), the output ELFs are compared
byte-for-byte, and each is run to a PREDICTED exit code (so the outputs are provably
real compilations, not a degenerate empty match). Byte-identical output on every
program is evidence the seed-built kovc behaves identically to the Python-built kovc
beyond the self-source -- within the i32 subset these programs exercise.

This file is the harness AND the corpus (programs are inline + auditable). It is the
backing artifact for the "Cross-check" section of docs/K_DDC_RESULT.md.

Run from the repo root:  python stage0/helixc-bootstrap/ddc_battery.py
(Both kovc binaries are LINUX ELFs; this script orchestrates WSL internally. It
builds K1py fresh and reuses /tmp/K1prime if present + correct size, else builds it.)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)
_UL = "ulimit -s unlimited 2>/dev/null || ulimit -s 1048576; "

# (name, source, predicted exit code mod 256). i32-only self-hosting subset.
CORPUS = [
    ("ret",         "fn main() -> i32 { 42 }", 42),
    ("mul",         "fn main() -> i32 { 6 * 7 }", 42),
    ("precedence",  "fn main() -> i32 { 2 + 3 * 4 }", 14),
    ("sub_assoc",   "fn main() -> i32 { 20 - 5 - 3 }", 12),
    ("div_assoc",   "fn main() -> i32 { 100 / 5 / 2 }", 10),
    ("modulo",      "fn main() -> i32 { 17 % 5 }", 2),
    ("bit_and",     "fn main() -> i32 { 12 & 10 }", 8),
    ("bit_or",      "fn main() -> i32 { 8 | 5 }", 13),
    ("bit_xor",     "fn main() -> i32 { 15 ^ 9 }", 6),
    ("signed_div",  "fn main() -> i32 { (0 - 6) / (0 - 2) }", 3),
    ("overflow",    "fn main() -> i32 { 65536 * 65536 }", 0),
    ("mut_local",   "fn main() -> i32 { let mut x = 41; x = x + 1; x }", 42),
    ("while_sum",   "fn main() -> i32 { let mut s = 0; let mut i = 1; while i <= 10 { s = s + i; i = i + 1 } s }", 55),
    ("six_cmp",     "fn main() -> i32 { (3 < 5) + (5 <= 5) + (9 > 2) + (2 >= 2) + (4 == 4) + (3 != 4) }", 6),
    ("nested_if",   "fn main() -> i32 { let x = 7; if x > 5 { if x > 9 { 1 } else { 100 } } else { 2 } }", 100),
    ("recursion",   "fn fac(n: i32) -> i32 { if n < 2 { 1 } else { n * fac(n - 1) } } fn main() -> i32 { fac(6) }", 208),
    ("fib",         "fn fib(n: i32) -> i32 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } } fn main() -> i32 { fib(13) }", 233),
    ("mutual_rec",  "fn is_even(n: i32) -> i32 { if n == 0 { 1 } else { is_odd(n - 1) } } fn is_odd(n: i32) -> i32 { if n == 0 { 0 } else { is_even(n - 1) } } fn main() -> i32 { is_even(10) }", 1),
    ("many_locals", "fn main() -> i32 { let a = 1; let b = 2; let c = 3; let d = 4; let e = 5; let f = 6; let g = 7; let h = 8; let i = 9; let j = 10; a + b + c + d + e + f + g + h + i + j }", 55),
    ("arena",       "fn main() -> i32 { __arena_push(10); __arena_push(20); __arena_push(25); let n = __arena_len(); __arena_get(0) + __arena_get(1) + __arena_get(2) + n }", 58),
    ("arena_set",   "fn main() -> i32 { __arena_push(10); __arena_push(20); __arena_set(0, 22); __arena_get(0) + __arena_get(1) }", 42),
]


def wsl(cmd, **kw):
    return subprocess.run(["wsl", "-e", "bash", "-c", cmd], **kw)


def sz(path):
    r = wsl(f"stat -c %s {path} 2>/dev/null || echo 0", capture_output=True, timeout=15)
    return int(r.stdout.decode().strip() or "0")


def build_compilers():
    """Build K1py fresh (Python route); reuse /tmp/K1prime if it is the known-good
    587092-byte seed mint, else build it with the seed (slow, ~4 min)."""
    subprocess.run([sys.executable, os.path.join(HERE, "assemble_k1.py")], check=True, timeout=120)
    driver = open(os.path.join(HERE, "k1src.hx"), encoding="utf-8").read()
    from helixc.tests.test_codegen import _compile_src_to_elf
    print("building K1py (Python reference) ...")
    k1py = _compile_src_to_elf(driver)
    wsl("cat > /tmp/K1py.bin", input=k1py, check=True, timeout=120)
    print(f"K1py = {len(k1py)} bytes")
    if sz("/tmp/K1prime") == 587092:
        print("reusing existing /tmp/K1prime (587092 bytes)")
    else:
        print("building /tmp/K1prime with the seed (~4 min) ...")
        wsl(f"cd {HERE.replace(chr(92), '/').replace('C:', '/mnt/c')} && rm -f /tmp/K1prime && ./seed.bin k1src.hx /tmp/K1prime",
            check=True, timeout=900)
    n = sz("/tmp/K1prime")
    print(f"K1prime = {n} bytes")
    return n > 0


def run_one(name, src, expected):
    """Compile `src` with BOTH kovc binaries, compare outputs, run the seed output."""
    wsl("cat > /tmp/k1_in.hx", input=src.encode("utf-8"), check=True, timeout=30)
    script = (
        _UL +
        "rm -f /tmp/k1_out.bin; chmod +x /tmp/K1prime; /tmp/K1prime >/dev/null 2>&1; "
        "cp /tmp/k1_out.bin /tmp/bat_seed 2>/dev/null; "
        "rm -f /tmp/k1_out.bin; chmod +x /tmp/K1py.bin; /tmp/K1py.bin >/dev/null 2>&1; "
        "cp /tmp/k1_out.bin /tmp/bat_py 2>/dev/null; "
        "echo SEED_MD5=$(md5sum /tmp/bat_seed 2>/dev/null | cut -d' ' -f1); "
        "echo PY_MD5=$(md5sum /tmp/bat_py 2>/dev/null | cut -d' ' -f1); "
        "if cmp -s /tmp/bat_seed /tmp/bat_py; then echo CMP=SAME; else echo CMP=DIFFER; fi; "
        "chmod +x /tmp/bat_seed 2>/dev/null; /tmp/bat_seed; echo EXIT=$?"
    )
    r = wsl(script, capture_output=True, timeout=120)
    out = r.stdout.decode()
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line and not line.startswith("/"))
    seed_md5 = kv.get("SEED_MD5", "")
    same = kv.get("CMP") == "SAME"
    try:
        got = int(kv.get("EXIT", "-1"))
    except ValueError:
        got = -1
    exit_ok = (got == (expected & 255))
    return same, exit_ok, got, seed_md5


def main():
    if not build_compilers():
        print("FAIL: could not build both compilers")
        return 3
    print(f"\n{'program':<13} {'cmp':<7} {'exit':>5} {'want':>5}  {'output md5':<32}")
    print("-" * 72)
    all_same = True
    all_exit = True
    distinct = set()
    for name, src, expected in CORPUS:
        same, exit_ok, got, md5 = run_one(name, src, expected)
        distinct.add(md5)
        all_same = all_same and same
        all_exit = all_exit and exit_ok
        flag = "" if (same and exit_ok) else "   <-- MISMATCH"
        print(f"{name:<13} {'SAME' if same else 'DIFFER':<7} {got:>5} {expected & 255:>5}  {md5:<32}{flag}")
    print("-" * 72)
    print(f"{len(CORPUS)} programs ; all byte-identical (K1' vs K1py): {all_same} ; "
          f"all exit codes match prediction: {all_exit} ; distinct output md5s: {len(distinct)}")
    if all_same and all_exit:
        print("\nBATTERY PASS: seed-built kovc and Python-built kovc produce byte-identical")
        print("ELFs on every program, each running to its predicted exit code. The seed's")
        print("fidelity to kovc holds well beyond the self-source (within the i32 subset).")
        return 0
    print("\nBATTERY FAIL: a program diverged or mis-ran -- inspect the MISMATCH rows above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
