"""K2.A (2026-05-26): parity harness scaffold.

The K2 milestone (per docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md) is a
test runner that, for every source program, compiles via BOTH paths
and compares output:

  - Python helixc path (compile_and_run from test_codegen)
  - Bootstrap kovc path (_kovc_self_host_compile_and_run from
    test_codegen)

For each source: both compilers, both binaries, same exit code =>
PASS. Different exit code => FAIL (parity violation).

This file ships the SCAFFOLD with a small starter corpus. Future
K2.* chunks expand the corpus to surface real-source parser/codegen
divergences that K1.* alone wouldn't find. The eventual goal is a
corpus large enough that "K2 green" is a credible gate for the
Python-ready-to-delete state.

Phase 1 (this file): behavioral parity (same exit code).
Phase 2 (future): structural parity (byte-identical ELF after
                  relocation normalization).
Phase 3 (future): N-generation fixpoint (kovc compiles kovc.hx to
                  kovc'; compile again to get kovc''; kovc' == kovc'').
"""
import os
import sys
import pytest

# Lazy imports inside the test fns so the file doesn't crash if WSL is
# degraded at collection time.


# The K2.A starter corpus. Each entry is (name, src, expected_rc).
#
# Selection criteria for K2.A:
#   - Each program is small and self-contained.
#   - Each program exercises a different category of feature so the
#     corpus has breadth even with only ~10 entries.
#   - Expected_rc is the value main() should return; both compilers
#     should agree.
#
# Future K2.* chunks expand this list (and eventually replace it with
# a directory of .hx files).
K2_CORPUS = [
    ("p01_basic_return",         "fn main() -> i32 { 42 }", 42),
    ("p02_let_int",              "fn main() -> i32 { let x = 42; x }", 42),
    ("p03_arith",                "fn main() -> i32 { 20 + 22 }", 42),
    ("p04_if_else",              "fn main() -> i32 { if true { 42 } else { 0 } }", 42),
    ("p05_match_lit",            "fn main() -> i32 { match 1 { 1 => 42, _ => 0 } }", 42),
    ("p06_for_loop_sum",         "fn main() -> i32 { let mut s = 0; for i in 0..7 { s = s + i; } s + 21 }", 42),
    ("p07_struct_lit",           "struct P { v: i32 } fn main() -> i32 { let p = P { v: 42 }; p.v }", 42),
    ("p08_tuple_field",          "fn main() -> i32 { let t = (42, 0); t.0 }", 42),
    ("p09_fn_call",              "fn f() -> i32 { 42 } fn main() -> i32 { f() }", 42),
    ("p10_neg_then_add",         "fn main() -> i32 { let x = -10 + 52; x }", 42),
]


@pytest.mark.parametrize("name,src,expected_rc", K2_CORPUS, ids=[c[0] for c in K2_CORPUS])
def test_k2_parity(name: str, src: str, expected_rc: int):
    """Run one K2 corpus entry through both compilers; assert
    behavioral parity.

    Two assertions:
      1. Python helixc rc == expected_rc (sanity: corpus item is correct).
      2. Bootstrap kovc rc == Python helixc rc (parity).

    If both pass, the corpus item is K2-clean. If (2) fails but (1)
    passes, the bootstrap diverges from Python helixc -- a parity
    violation surfacing a K1 gap that this corpus item caught.
    """
    # Defer imports until test execution (so a broken Python helixc
    # doesn't blow up the test collection step).
    from helixc.tests.test_codegen import (
        compile_and_run as python_compile_and_run,
        _kovc_self_host_compile_and_run as bootstrap_compile_and_run,
    )

    # Path 1: Python helixc.
    python_rc = python_compile_and_run(src, optimize=True)
    assert python_rc == expected_rc, (
        f"K2.A corpus {name}: Python helixc rc={python_rc}, "
        f"expected {expected_rc}. Corpus item is wrong, or Python "
        f"helixc regressed."
    )

    # Path 2: Bootstrap kovc self-host.
    bootstrap_rc = bootstrap_compile_and_run(f"k2_parity_{name}", src)

    # Parity assertion: both compilers must agree.
    assert bootstrap_rc == python_rc, (
        f"K2.A parity violation on {name}: "
        f"Python helixc rc={python_rc}, bootstrap kovc rc={bootstrap_rc}. "
        f"Behavioral divergence -- a K1 chunk is missing for this shape."
    )


def test_k2_corpus_size():
    """Sanity check: corpus has at least 10 entries at K2.A.

    Future K2.* chunks expand the corpus. This test guards against
    accidental shrinkage. The growth ratchet is one-way: each K2.*
    chunk strictly increases the lower bound (e.g., K2.B might bump
    to 20, K2.C to 40, etc.) until a credible "K2 green over a real-
    source corpus" threshold is reached.
    """
    assert len(K2_CORPUS) >= 10, (
        f"K2.A corpus shrank to {len(K2_CORPUS)} entries. The K2 "
        f"growth ratchet is one-way -- entries can be replaced but "
        f"not net-removed."
    )
