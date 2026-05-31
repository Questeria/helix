#!/usr/bin/env python3
"""Diverse-double-compile (DDC) trust check for the helixc-bootstrap seed.

The point of the seed is to retire Python from the trust chain. DDC proves the
seed did so *faithfully* -- that it reproduced the real `kovc` compiler and did
not inject anything (a la Thompson's "Reflections on Trusting Trust"). Wheeler's
diverse-double-compiling: build the compiler two INDEPENDENT ways and compare
their *output* at the self-hosting fixpoint; the two routes' own codegen
differences wash out, so a byte-identical match proves semantic equivalence.

    Route A (seed):    k1src.hx --seed (stage0 ladder, C)--> K1'   --(BIG)--> K2_seed
    Route B (Python):  k1src.hx --python reference compiler--> K1py  --(BIG)--> K2_python

K1' and K1py differ in bytes (different compilers built kovc) -- that is
expected and fine. What must match is what they PRODUCE: feed both the SAME
1.5 MB compiler source (BIG = k1input.hx) and compare K2_seed vs K2_python.

    K2_seed == K2_python  <=>  the seed built a faithful kovc  =>  Python is provably redundant.

K1' (the seed route) is built separately by the seed binary (slow, O(n^2)); this
script locates it at /tmp/K1prime (override with argv[1]), builds the Python
route here, runs both under a big stack, and compares. Everything runs in ONE
process so WSL /tmp state cannot be cleared between steps.

Run from the repo root:  python stage0/helixc-bootstrap/ddc_check.py [K1prime_path]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)

# The bootstrap parser is deeply recursive (parse_primary has ~1241 lets), which
# overflows the default 8 MB stack on the full self-compile -- same reason the
# canonical self-host test uses an unlimited stack.
_UL = "ulimit -s unlimited 2>/dev/null || ulimit -s 1048576; "


def wsl(cmd, **kw):
    return subprocess.run(["wsl", "-e", "bash", "-c", cmd], **kw)


def sz(path):
    r = wsl(f"stat -c %s {path} 2>/dev/null || echo 0", capture_output=True, timeout=15)
    return int(r.stdout.decode().strip() or "0")


def main():
    k1prime = sys.argv[1] if len(sys.argv) > 1 else "/tmp/K1prime"
    driver = open(os.path.join(HERE, "k1src.hx"), encoding="utf-8").read()    # the compiler source
    big = open(os.path.join(HERE, "k1input.hx"), encoding="utf-8").read()     # 1.5 MB self-source (BIG)
    print(f"driver = {len(driver.encode('utf-8'))} bytes ; BIG = {len(big.encode('utf-8'))} bytes ; K1' = {k1prime}")

    # ---- Route B (Python): build K1py, run it on BIG -> K2_python ----
    from helixc.tests.test_codegen import _compile_src_to_elf
    print("[B] building K1py via the Python reference compiler ...")
    k1py = _compile_src_to_elf(driver)
    wsl("cat > /tmp/K1py.bin", input=k1py, check=True, timeout=120)
    print(f"[B] K1py = {len(k1py)} bytes ; compiling BIG -> K2_python ...")
    wsl("cat > /tmp/k1_in.hx", input=big.encode("utf-8"), check=True, timeout=120)
    wsl(f"rm -f /tmp/k1_out.bin; {_UL}chmod +x /tmp/K1py.bin && /tmp/K1py.bin", timeout=600)
    wsl("cp /tmp/k1_out.bin /tmp/K2_python 2>/dev/null", timeout=15)
    n_py = sz("/tmp/K2_python")
    print(f"[B] K2_python = {n_py} bytes")
    if n_py == 0:
        print("FAIL: the Python route did not produce K2_python")
        return 3

    # ---- Route A (seed): run the seed-built K1' on BIG -> K2_seed ----
    if sz(k1prime) == 0:
        print(f"PENDING: K1' not found at {k1prime} -- build it with the seed, then re-run.")
        return 2
    print(f"[A] compiling BIG with the seed-built K1' -> K2_seed ...")
    wsl("cat > /tmp/k1_in.hx", input=big.encode("utf-8"), check=True, timeout=120)
    wsl(f"rm -f /tmp/k1_out.bin; {_UL}chmod +x {k1prime} && {k1prime}", timeout=600)
    wsl("cp /tmp/k1_out.bin /tmp/K2_seed 2>/dev/null", timeout=15)
    n_seed = sz("/tmp/K2_seed")
    print(f"[A] K2_seed = {n_seed} bytes")
    if n_seed == 0:
        print("FAIL: the seed route did not produce K2_seed")
        return 3

    # ---- Compare the two routes at the fixpoint ----
    cmp = wsl("cmp -s /tmp/K2_seed /tmp/K2_python && echo IDENTICAL || echo DIFFER",
              capture_output=True, timeout=30)
    verdict = cmp.stdout.decode().strip()
    print(f"\nK2_seed vs K2_python : {verdict}  ({n_seed} vs {n_py} bytes)")

    # ---- Bonus: prove K2_seed is a WORKING compiler (compile 6*7 -> exit 42) ----
    # BIG's main reads /tmp/k2_in.hx and writes /tmp/k2_out.bin, so K2_seed does too.
    wsl('printf "%s" "fn main() -> i32 { 6 * 7 }" > /tmp/k2_in.hx', check=True, timeout=15)
    wsl(f"rm -f /tmp/k2_out.bin; {_UL}chmod +x /tmp/K2_seed && /tmp/K2_seed", timeout=120)
    r = wsl("chmod +x /tmp/k2_out.bin 2>/dev/null && /tmp/k2_out.bin; echo exit=$?",
            capture_output=True, timeout=30)
    k2_works = b"exit=42" in r.stdout
    print(f"K2_seed compiles 6*7 -> {'exit 42 (WORKS)' if k2_works else 'FAILED: ' + r.stdout.decode().strip()}")

    if verdict == "IDENTICAL" and k2_works:
        print("\nDDC PASS: the seed route and the Python route converge byte-for-byte at the")
        print("self-hosting fixpoint, and the seed-built compiler works. The seed faithfully")
        print("reproduced kovc -- Python is provably redundant in the trust chain.")
        return 0
    print("\nDDC did NOT pass cleanly -- inspect K2_seed vs K2_python above. Do NOT claim trust.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
