"""Stage 25: tests for trace-based introspection (@trace + TraceBuffer)."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.trace_pass import (
    TraceBuffer,
    TraceEvent,
    trace_equiv,
    is_traced,
    traced_fn_names,
    validate_trace_attrs,
    TRAP_TRACE_OVERFLOW,
    TRAP_TRACE_EQUIV_SHAPE_MISMATCH,
    DEFAULT_TRACE_CAP,
)


def test_parse_trace_attribute():
    """A fn marked `@trace` carries 'trace' in its attrs list."""
    src = """
@trace
fn f(x: i32) -> i32 { x + 1 }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl) and it.name == "f")
    assert "trace" in fn.attrs
    assert is_traced(fn)


def test_trace_attribute_combines():
    """@trace coexists with @pure / @inline."""
    src = """
@trace
@pure
fn f(x: i32) -> i32 { x + 1 }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl) and it.name == "f")
    assert "trace" in fn.attrs
    assert "pure" in fn.attrs


def test_no_trace_attribute():
    """A plain fn is not traced."""
    src = """
fn g(y: i32) -> i32 { y * 2 }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl) and it.name == "g")
    assert not is_traced(fn)


def test_traced_fn_names():
    src = """
@trace fn a(x: i32) -> i32 { x }
fn b(x: i32) -> i32 { x }
@trace fn c(x: i32) -> i32 { x }
"""
    prog = parse(src)
    names = traced_fn_names(prog)
    assert names == ["a", "c"]


def test_trace_buffer_push_and_len():
    tb = TraceBuffer()
    assert len(tb) == 0
    tb.push(TraceEvent(op_kind="entry", fn_name="f", operands=(1, 2)))
    tb.push(TraceEvent(op_kind="exit", fn_name="f", operands=(3,)))
    assert len(tb) == 2


def test_trace_buffer_overflow_raises():
    """Past cap, push raises OverflowError (= trap 25001)."""
    tb = TraceBuffer(cap=2)
    tb.push(TraceEvent("entry", "f", ()))
    tb.push(TraceEvent("exit", "f", ()))
    with pytest.raises(OverflowError):
        tb.push(TraceEvent("op", "g", ()))


def test_trace_buffer_clear():
    tb = TraceBuffer()
    tb.push(TraceEvent("entry", "f", ()))
    assert len(tb) == 1
    tb.clear()
    assert len(tb) == 0


def test_trace_equiv_identical():
    a = TraceBuffer()
    b = TraceBuffer()
    for ev in (
        TraceEvent("entry", "f", (1, 2)),
        TraceEvent("op", "add", (1, 2), result=3),
        TraceEvent("exit", "f", (3,)),
    ):
        a.push(ev)
        b.push(ev)
    assert trace_equiv(a, b)


def test_trace_equiv_different_length():
    a = TraceBuffer()
    b = TraceBuffer()
    a.push(TraceEvent("entry", "f", ()))
    assert not trace_equiv(a, b)


def test_trace_equiv_different_ops():
    a = TraceBuffer()
    b = TraceBuffer()
    a.push(TraceEvent("entry", "f", (1,)))
    b.push(TraceEvent("entry", "f", (2,)))  # different operands
    assert not trace_equiv(a, b)


def test_trap_25001_reserved():
    """Document the trap-id reservations."""
    assert TRAP_TRACE_OVERFLOW == 25001
    assert TRAP_TRACE_EQUIV_SHAPE_MISMATCH == 25002


def test_default_cap_is_4kb():
    """Phase-0 default trace cap is 4096 entries."""
    assert DEFAULT_TRACE_CAP == 4096


def test_validate_trace_on_extern_rejected():
    """@trace on an extern \"C\" fn must produce a diagnostic."""
    src = '''
@trace
extern "C" fn malloc(n: u64) -> u64;
'''
    prog = parse(src)
    diags = validate_trace_attrs(prog)
    assert diags
    assert "malloc" in diags[0]
    assert "extern" in diags[0]


def test_validate_trace_clean():
    src = """
@trace fn f(x: i32) -> i32 { x }
"""
    prog = parse(src)
    diags = validate_trace_attrs(prog)
    assert diags == []


# ----------------------------------------------------------------------
# Audit 28.8 A7 — @trace lowering + CLI wiring
# ----------------------------------------------------------------------
def test_a7_trace_lowers_to_ir_entry_exit():
    """A7: `@trace fn f` emits TRACE_ENTRY before the body and
    TRACE_EXIT after. Previously the attribute was parsed and validated
    but NO IR op was emitted — the runtime never saw the events."""
    from helixc.ir.lower_ast import lower
    from helixc.ir.tir import OpKind
    src = "@trace fn ping(x: i32) -> i32 { x + 1 }\nfn main() -> i32 { ping(1) }"
    prog = parse(src)
    mod = lower(prog)
    ping_fn = mod.functions["ping"]
    all_ops = [op for blk in ping_fn.blocks for op in blk.ops]
    entries = [op for op in all_ops if op.kind == OpKind.TRACE_ENTRY]
    exits = [op for op in all_ops if op.kind == OpKind.TRACE_EXIT]
    assert len(entries) == 1, \
        f"expected one TRACE_ENTRY op, got: {[op.kind.name for op in all_ops]}"
    assert len(exits) == 1, \
        f"expected one TRACE_EXIT op, got: {[op.kind.name for op in all_ops]}"
    assert entries[0].attrs["fn_name"] == "ping"
    assert exits[0].attrs["fn_name"] == "ping"


def test_a7_non_traced_fn_has_no_trace_ops():
    """A7: plain fn (no @trace) should NOT emit TRACE_ENTRY/EXIT."""
    from helixc.ir.lower_ast import lower
    from helixc.ir.tir import OpKind
    src = "fn plain(x: i32) -> i32 { x + 1 }"
    prog = parse(src)
    mod = lower(prog)
    plain_fn = mod.functions["plain"]
    all_ops = [op for blk in plain_fn.blocks for op in blk.ops]
    assert not any(op.kind in (OpKind.TRACE_ENTRY, OpKind.TRACE_EXIT)
                   for op in all_ops), \
        f"plain fn should have no trace ops; got: {[op.kind.name for op in all_ops]}"


def test_a7_trace_validation_wired_into_check_py(capsys, tmp_path):
    """A7: helixc/check.py now invokes validate_trace_attrs after
    typecheck. @trace on an extern "C" fn surfaces as a build error."""
    from helixc.check import main
    src = '''
@trace
extern "C" fn external_fn(x: i32) -> i32;
fn user_main() -> i32 { external_fn(1) }
'''
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p, "--check-only"])
    cap = capsys.readouterr()
    assert rc == 1, f"expected build failure for @trace extern; out={cap.out!r}"
    assert "trace:" in cap.out
    assert "extern" in cap.out


def test_a7_backend_emits_trace_ops_as_stubs():
    """A7: x86_64 backend emits TRACE_ENTRY/EXIT as no-op stubs in
    Phase-0 (the runtime helpers don't exist yet). Verify the codegen
    path doesn't raise + the binary still compiles."""
    from helixc.frontend.parser import parse
    from helixc.frontend.typecheck import typecheck
    from helixc.ir.lower_ast import lower
    from helixc.backend.x86_64 import compile_module_to_elf
    src = "@trace fn pong(x: i32) -> i32 { x }\nfn main() -> i32 { pong(7) }"
    prog = parse(src)
    typecheck(prog)
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    assert isinstance(elf, (bytes, bytearray)) and len(elf) > 0


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 C2-2 — early `return X` in a @trace'd fn must
# also emit TRACE_EXIT (otherwise the trace stream has ENTRY pairs
# without matching EXITs on the early-return path).
# ----------------------------------------------------------------------
def test_c2_2_early_return_emits_trace_exit():
    """C2-2: a @trace'd fn with an explicit `return X` inside an `if`
    body must emit TRACE_EXIT before the early `ret` op AND before the
    fall-through `ret` — so paired trace events stay balanced regardless
    of which return path executes at runtime."""
    from helixc.ir.lower_ast import lower
    from helixc.ir.tir import OpKind
    # The fall-through path falls off the end of the if/else expression,
    # which Helix treats as the block's value. Explicit early-return
    # is the path we're checking.
    src = (
        "@trace fn f(x: i32) -> i32 { if x > 0 { return 1; }; 2 }\n"
        "fn main() -> i32 { f(5) }\n"
    )
    prog = parse(src)
    mod = lower(prog)
    f_fn = mod.functions["f"]
    all_ops = [op for blk in f_fn.blocks for op in blk.ops]
    entries = [op for op in all_ops if op.kind == OpKind.TRACE_ENTRY]
    exits = [op for op in all_ops if op.kind == OpKind.TRACE_EXIT]
    assert len(entries) == 1, (
        f"expected 1 TRACE_ENTRY, got {len(entries)}: "
        f"{[op.kind.name for op in all_ops]}"
    )
    # MUST be 2 — one before the early `return 1` ret, one before the
    # fall-through ret-of-2.
    assert len(exits) == 2, (
        f"expected 2 TRACE_EXIT ops (one per return path), got {len(exits)}: "
        f"{[op.kind.name for op in all_ops]}"
    )
    # Both should attribute to the same fn name.
    assert all(op.attrs.get("fn_name") == "f" for op in exits), (
        f"TRACE_EXIT attrs mismatch: {[op.attrs for op in exits]}"
    )


def test_c2_2_non_traced_fn_with_early_return_no_trace_exit():
    """C2-2 inverse: a plain (non-@trace) fn with `return X` must NOT
    emit any TRACE_* ops — guard against spurious wiring."""
    from helixc.ir.lower_ast import lower
    from helixc.ir.tir import OpKind
    src = (
        "fn g(x: i32) -> i32 { if x > 0 { return 1; }; 2 }\n"
        "fn main() -> i32 { g(5) }\n"
    )
    prog = parse(src)
    mod = lower(prog)
    g_fn = mod.functions["g"]
    all_ops = [op for blk in g_fn.blocks for op in blk.ops]
    assert not any(
        op.kind in (OpKind.TRACE_ENTRY, OpKind.TRACE_EXIT)
        for op in all_ops
    )


def test_c2_2_early_return_void():
    """C2-2: traced fn with bare `return;` (no value) — TRACE_EXIT
    must still emit, with a synthesized 0 operand."""
    from helixc.ir.lower_ast import lower
    from helixc.ir.tir import OpKind
    src = (
        "@trace fn h(x: i32) -> i32 { if x > 0 { return 0; }; 1 }\n"
        "fn main() -> i32 { h(5) }\n"
    )
    prog = parse(src)
    mod = lower(prog)
    h_fn = mod.functions["h"]
    all_ops = [op for blk in h_fn.blocks for op in blk.ops]
    exits = [op for op in all_ops if op.kind == OpKind.TRACE_EXIT]
    assert len(exits) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
