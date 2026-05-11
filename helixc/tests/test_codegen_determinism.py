"""Stage 28.8.1 — codegen determinism byte-identical regression tests.

Background: cycle 11 silent-failures audit
(``docs/audit-stage28-8-cycle11-silent-failures.md`` lines 412-486)
documented that ``test_bootstrap_kovc_full_pipeline_arithmetic``
produced three distinct cache hashes across three invocations of the
same source. The pre-Phase-A finalization research traced the cause
to:

  1. ``backend/x86_64.py`` used ``f"...{id(op):x}"`` at 9 call sites
     for symbol generation — ``id()`` is the Python object's memory
     address, which varies across processes.

  2. ``frontend/match_lower.py``'s ``_FRESH_COUNTER`` was module-level
     mutable state never reset between ``lower_matches()`` calls.

These tests assert that the fixes hold: compiling the same source
multiple times in the same process (and across subprocesses) produces
byte-identical ELF output. Stage 29's byte-identical-verification
gate depends on this.

License: Apache 2.0
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from helixc.backend.x86_64 import compile_module_to_elf
from helixc.frontend.flatten_impls import flatten_impls
from helixc.frontend.flatten_modules import flatten_modules
from helixc.frontend.grad_pass import grad_pass
from helixc.frontend.match_lower import _FRESH_COUNTER, lower_matches
from helixc.frontend.monomorphize import monomorphize
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.cse import cse_module
from helixc.ir.passes.dce import dce_module
from helixc.ir.passes.fdce import fdce_module


def _compile_to_elf_bytes(src: str) -> bytes:
    """Mirror of ``test_codegen.compile_and_run``'s pipeline up to ELF
    emission. Each invocation reparses + rebuilds the IR + recompiles
    from scratch — no shared state between runs."""
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


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# In-process determinism
# ---------------------------------------------------------------------------

def test_codegen_determinism_byte_identical_simple():
    """Stage 28.8.1: same minimal source -> identical ELF bytes 5x."""
    src = "fn main() -> i32 { 42 }"
    hashes = {_sha256(_compile_to_elf_bytes(src)) for _ in range(5)}
    assert len(hashes) == 1, (
        f"expected 1 unique hash across 5 compilations, got {len(hashes)}: "
        f"{hashes}"
    )


def test_codegen_determinism_byte_identical_with_print():
    """Stage 28.8.1: PRINT ops were a major id(op) leak site (`sym
    = f"__helix_str_{id(op):x}"`); compile a program that exercises
    them and assert byte-identity 5x."""
    src = 'fn main() -> i32 { print_str("hi"); 0 }'
    hashes = {_sha256(_compile_to_elf_bytes(src)) for _ in range(5)}
    assert len(hashes) == 1, (
        f"expected 1 unique hash across 5 compilations, got {len(hashes)}: "
        f"{hashes}"
    )


def test_codegen_determinism_byte_identical_with_panic():
    """Stage 28.8.1: TRAP ops embed an id(op)-suffixed symbol; assert
    byte-identity 5x for a panic-containing program."""
    src = 'fn main() -> i32 { panic("nope"); 0 }'
    hashes = {_sha256(_compile_to_elf_bytes(src)) for _ in range(5)}
    assert len(hashes) == 1, (
        f"expected 1 unique hash across 5 compilations, got {len(hashes)}: "
        f"{hashes}"
    )


def test_codegen_determinism_with_match():
    """Stage 28.8.1: match_lower's _FRESH_COUNTER was the second
    determinism leak. Compile a match-using program 5x and assert
    byte-identity."""
    src = (
        "fn main() -> i32 {\n"
        "    let x = 3;\n"
        "    match x { 0 => 100, 1 => 200, _ => 42 }\n"
        "}\n"
    )
    hashes = {_sha256(_compile_to_elf_bytes(src)) for _ in range(5)}
    assert len(hashes) == 1, (
        f"expected 1 unique hash across 5 compilations, got {len(hashes)}: "
        f"{hashes}"
    )


# ---------------------------------------------------------------------------
# match_lower fresh counter reset
# ---------------------------------------------------------------------------

def test_match_lower_fresh_counter_resets_per_call():
    """Stage 28.8.1: `lower_matches(prog)` must reset `_FRESH_COUNTER`
    so the synthesized scrutinee names are identical across calls.

    Pre-fix: two calls on the same source produced `__scrut_1` then
    `__scrut_2` (or `__scrut_47` if other tests ran first), polluting
    the IR with order-dependent names.
    """
    src = "fn main() -> i32 { match 3 { 0 => 1, _ => 2 } }"

    def _names(s: str) -> list[str]:
        prog = parse(s, include_stdlib=False)
        lower_matches(prog)
        # Scan the rewritten program for any `__scrut_*` names.
        out: list[str] = []
        for item in prog.items:
            txt = repr(item)
            for tok in txt.split():
                if "__scrut_" in tok:
                    out.append(tok)
        return out

    first = _names(src)
    second = _names(src)
    assert first == second, (
        f"_FRESH_COUNTER was not reset: first={first} second={second}"
    )


def test_match_lower_fresh_counter_state_visible():
    """Stage 28.8.1: also assert that _FRESH_COUNTER is observably
    reset to 0 at the start of lower_matches and incremented from
    there. This guards against accidental refactors that drop the
    reset."""
    # Pollute the counter from an unrelated call sequence.
    _FRESH_COUNTER[0] = 1000
    src = "fn main() -> i32 { match 0 { _ => 1 } }"
    prog = parse(src, include_stdlib=False)
    lower_matches(prog)
    # After lower_matches, the counter should be small (started at 0,
    # incremented once per fresh name). Specifically the pre-fix
    # 1000+ ceiling no longer applies.
    assert _FRESH_COUNTER[0] < 100, (
        f"_FRESH_COUNTER not reset; observed value {_FRESH_COUNTER[0]} "
        f"after a single match lowering"
    )


# ---------------------------------------------------------------------------
# Cross-subprocess determinism
# ---------------------------------------------------------------------------

def test_codegen_determinism_subprocess():
    """Stage 28.8.1: compile the same source from N=3 separate Python
    subprocesses; all produced ELF bytes must be identical. This is
    the strictest gate — defeats any process-address leak (id() / hash
    randomization / dict iteration order)."""
    src = 'fn main() -> i32 { let x = 21; x + x }'
    script = (
        "import sys, hashlib, os\n"
        "sys.path.insert(0, "
        + repr(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
        + ")\n"
        "from helixc.frontend.parser import parse\n"
        "from helixc.frontend.grad_pass import grad_pass\n"
        "from helixc.frontend.monomorphize import monomorphize\n"
        "from helixc.frontend.flatten_modules import flatten_modules\n"
        "from helixc.frontend.flatten_impls import flatten_impls\n"
        "from helixc.ir.lower_ast import lower\n"
        "from helixc.ir.passes.const_fold import fold_module\n"
        "from helixc.ir.passes.cse import cse_module\n"
        "from helixc.ir.passes.dce import dce_module\n"
        "from helixc.ir.passes.fdce import fdce_module\n"
        "from helixc.backend.x86_64 import compile_module_to_elf\n"
        "prog = parse(" + repr(src) + ", include_stdlib=True)\n"
        "flatten_modules(prog); flatten_impls(prog); monomorphize(prog)\n"
        "grad_pass(prog)\n"
        "mod = lower(prog); fold_module(mod); cse_module(mod); dce_module(mod); fdce_module(mod)\n"
        "elf = compile_module_to_elf(mod)\n"
        "print(hashlib.sha256(elf).hexdigest())\n"
    )
    hashes: set[str] = set()
    for _ in range(3):
        # Stage 28.8.1: subprocess invocation isolates Python's id()
        # allocations, hash seeds, and any module-level state. If the
        # output bytes differ across subprocesses, a non-determinism
        # source remains.
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"subprocess failed: stderr=\n{result.stderr}"
        )
        hashes.add(result.stdout.strip())
    assert len(hashes) == 1, (
        f"cross-subprocess byte-identity violated: {len(hashes)} hashes "
        f"observed: {hashes}"
    )
