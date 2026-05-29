"""v3.1 step 6a — pins the shared-constants module.

`helixc/backend/_shared_constants.py` is the single source of truth
for runtime layout (reflection cells, arena cap, trace buffer cap,
SysV ABI). Both backends import from it; lower_ast.py reads the
cell count from it. These tests pin:

- The constants have the well-known values (a bump here is a
  user-observable cap change — the test failure surface forces the
  V3_HANDOFF doc update).
- x86_64.py and llvm_ir.py read from the shared module, not from
  inline private duplicates (drift surface eliminated).
"""
from __future__ import annotations

from helixc.backend import _shared_constants as shared


# --------------------------------------------------------------------------
# Pinned values
# --------------------------------------------------------------------------
def test_reflection_cell_count_pinned():
    """64 mutable reflection cells. A bump is a public ABI change."""
    assert shared.HELIX_NUM_CELLS == 64
    assert shared.HELIX_CELL_SIZE == 8


def test_arena_cap_pinned():
    """6 291 456 i32 slots = 24 MB BSS arena. Rescaled 3x from 2097152 on
    2026-05-28 (user-approved) so the self-built compiler can compile its
    own ~1.4 MB source (~2.72M slots) without arena overflow. A bump is a
    public ABI change -- must stay equal to kovc.hx helix_arena_cap()."""
    assert shared.HELIX_ARENA_CAP == 6291456


def test_trace_cap_pinned():
    """1024 trace events = 8 KB BSS ring buffer."""
    assert shared.HELIX_TRACE_CAP == 1024


def test_sysv_stack_layout_pinned():
    """SysV ABI: 16-byte alignment, 16-byte stack-arg base, 8-byte
    stride. A bump on any of these is a callee/caller-mismatch
    risk — must be tested explicitly."""
    assert shared.SYSV_STACK_ARG_BASE == 16
    assert shared.SYSV_STACK_ARG_STRIDE == 8
    assert shared.SYSV_STACK_ALIGNMENT == 16


# --------------------------------------------------------------------------
# Backends read from the shared module (single source of truth pin)
# --------------------------------------------------------------------------
def test_x86_backend_reexports_from_shared():
    """`helixc.backend.x86_64` re-exports the constants so legacy
    `from helixc.backend.x86_64 import HELIX_NUM_CELLS` still
    works.

    The `is` check on HELIX_ARENA_CAP (2_097_152) is the LOAD-
    BEARING identity canary — CPython caches small-int literals
    (-5..256, and on 3.13+ the co_consts free-list also re-uses
    literal slots), so `is` for HELIX_NUM_CELLS (64) etc. could
    silently pass even if x86_64.py had `HELIX_NUM_CELLS = 64`
    re-binding. ARENA_CAP is well outside any cache, so any
    drift would surface as `is` False.

    The source-grep check below is the belt-and-braces guard
    for the cache-range constants — it forbids a literal
    re-assignment `HELIX_NUM_CELLS = <num>` in x86_64.py."""
    from helixc.backend import x86_64
    # All values match — but `is` is only diagnostic for ARENA_CAP.
    assert x86_64.HELIX_NUM_CELLS == shared.HELIX_NUM_CELLS
    assert x86_64.HELIX_CELL_SIZE == shared.HELIX_CELL_SIZE
    # Load-bearing identity canary.
    assert x86_64.HELIX_ARENA_CAP is shared.HELIX_ARENA_CAP
    assert x86_64.HELIX_TRACE_CAP == shared.HELIX_TRACE_CAP
    assert x86_64.SYSV_STACK_ARG_BASE == shared.SYSV_STACK_ARG_BASE
    assert x86_64.SYSV_STACK_ARG_STRIDE == shared.SYSV_STACK_ARG_STRIDE
    assert x86_64.SYSV_STACK_ALIGNMENT == shared.SYSV_STACK_ALIGNMENT
    # Belt-and-braces: forbid direct `<NAME> = <literal>` re-binding
    # in x86_64.py for any of the shared constants. A drift back to
    # an inline definition would be caught here regardless of int
    # cache behavior.
    import inspect
    import re
    x86_src = inspect.getsource(x86_64)
    for name in (
            "HELIX_NUM_CELLS", "HELIX_CELL_SIZE", "HELIX_ARENA_CAP",
            "HELIX_TRACE_CAP", "SYSV_STACK_ARG_BASE",
            "SYSV_STACK_ARG_STRIDE", "SYSV_STACK_ALIGNMENT"):
        # A top-of-line `<NAME> = <something>` would be a re-binding.
        # The re-export `from ._shared_constants import (NAME, ...)`
        # has `NAME,` (no `=`) so this won't false-positive.
        pat = rf"^{name}\s*="
        assert not re.search(pat, x86_src, re.MULTILINE), (
            f"x86_64.py contains a direct `{name} = ...` assignment; "
            f"the constant must come from _shared_constants via "
            f"`from ._shared_constants import` only.")


def test_llvm_backend_reads_from_shared():
    """`helixc.backend.llvm_ir` re-aliases the relevant runtime
    constants under underscored names. Pin (a) value equality and
    (b) the source grep — the underscored alias must be assigned
    from `_shared.<NAME>`, not from a literal."""
    from helixc.backend import llvm_ir
    # Value equality plus the identity canary on ARENA_CAP.
    assert llvm_ir._HELIX_NUM_CELLS == shared.HELIX_NUM_CELLS
    assert llvm_ir._HELIX_ARENA_CAP is shared.HELIX_ARENA_CAP
    assert llvm_ir._HELIX_TRACE_CAP == shared.HELIX_TRACE_CAP
    # Belt-and-braces: forbid `_HELIX_NUM_CELLS = <numeric literal>`
    # in llvm_ir.py. The post-6a binding is
    # `_HELIX_NUM_CELLS = _shared.HELIX_NUM_CELLS` (= followed by
    # the shared-module accessor, NOT a numeric literal).
    import inspect
    import re
    llvm_src = inspect.getsource(llvm_ir)
    for name in ("_HELIX_NUM_CELLS", "_HELIX_ARENA_CAP", "_HELIX_TRACE_CAP"):
        # Look for `<NAME> = <integer-literal>` at line start —
        # i.e., a regression to the pre-6a inline definition. Allow
        # `<NAME> = _shared.<...>` (the post-6a binding).
        pat = rf"^{name}\s*=\s*\d"
        assert not re.search(pat, llvm_src, re.MULTILINE), (
            f"llvm_ir.py contains a direct `{name} = <literal>` "
            f"assignment; the alias must be sourced from "
            f"_shared_constants only.")


def test_shared_module_symbols_all_reexported_by_x86():
    """v3.1 step 6a mirror contract: every PUBLIC symbol in
    `_shared_constants` (anything not prefixed with `_`) must be
    re-exported by `x86_64.py`. A future shared-only addition
    that isn't mirrored would break callers that
    `from helixc.backend.x86_64 import <new_const>`.

    This is a forward-looking guard, not a current-state check —
    today's 7 constants are all mirrored. Future additions must
    update both the shared module AND the x86_64 re-export list."""
    from helixc.backend import x86_64
    shared_public = {
        n for n in dir(shared) if not n.startswith("_")}
    x86_public = {n for n in dir(x86_64) if not n.startswith("_")}
    missing = shared_public - x86_public
    assert not missing, (
        f"_shared_constants exports {missing!r} that are not "
        f"re-exported from x86_64.py — add them to the "
        f"`from ._shared_constants import (...)` block.")


def test_lower_ast_reads_cell_count_from_shared():
    """The frontend's quote()-lowering uses HELIX_NUM_CELLS to bound
    the distinct-AST cell-handle table. It MUST read from the
    shared module (not directly from x86_64) so a v3.1 step 6 x86
    deletion does not break it.

    Grep the lower_ast source for the canonical import line — a
    drift back to `from ..backend.x86_64 import HELIX_NUM_CELLS`
    would be a regression."""
    import inspect
    from helixc.ir import lower_ast
    src = inspect.getsource(lower_ast)
    assert (
        "from ..backend._shared_constants import HELIX_NUM_CELLS"
        in src), (
        "lower_ast.py must import HELIX_NUM_CELLS from "
        "_shared_constants, not from the x86_64 backend")
    assert (
        "from ..backend.x86_64 import HELIX_NUM_CELLS"
        not in src), (
        "lower_ast.py must NOT import HELIX_NUM_CELLS from "
        "the x86_64 backend (drift surface during v3.1 cutover)")
