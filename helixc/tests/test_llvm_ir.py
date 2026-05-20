"""Tests for helixc.backend.llvm_ir — v3.0 Phase D Stage 200
(the additive LLVM IR emitter substrate; scalar core only).

The emitter consumes the same host IR (`tir.Module`) that
`x86_64.py::compile_module_to_elf` consumes. Stage 200 covers the
scalar core: module header / target triple, `define`, integer
constants, integer add/sub/mul, `ret`. Everything else must be
REJECTED loudly with `LLVMEmitError` — a partial backend fails
closed, it never emits wrong IR.
"""

from __future__ import annotations

import pytest

from helixc.ir import tir
from helixc.backend import llvm_ir


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _i32() -> tir.TIRScalar:
    return tir.TIRScalar("i32")


# --------------------------------------------------------------------------
# scalar core — happy path
# --------------------------------------------------------------------------
def test_stage200_emit_const_return():
    """`fn main() -> i32 { 42 }` -> a `define` whose sole instruction is
    `ret i32 42` (the CONST_INT is materialized as an inline literal)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    b.ret(b.const_int(42))
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    assert 'target triple = "x86_64-unknown-linux-gnu"' in ll
    assert "define i32 @main() {" in ll
    assert "ret i32 42" in ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_emit_add():
    """CONST_INT + CONST_INT -> `%vN = add i32 2, 3` with the constants
    inlined; the ADD result is returned."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    s = b.add(b.const_int(2), b.const_int(3))
    b.ret(s)
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    assert f"%v{s.id} = add i32 2, 3" in ll, ll
    assert f"ret i32 %v{s.id}" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_emit_sub_and_mul():
    """SUB and MUL lower to the LLVM `sub` / `mul` mnemonics."""
    for kind, mnemonic in ((tir.OpKind.SUB, "sub"), (tir.OpKind.MUL, "mul")):
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        a = b.const_int(9)
        c = b.const_int(4)
        r = b.emit(kind, a, c, result_ty=_i32())
        b.ret(r)
        b.end_function()

        ll = llvm_ir.emit_module(mod)
        assert f"%v{r.id} = {mnemonic} i32 9, 4" in ll, (mnemonic, ll)
        assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_emit_function_with_params():
    """Parameters become `%v<id>` registers in the `define` signature
    and are referenced by name in the body."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "addp", [("a", _i32()), ("b", _i32())], _i32())
    s = b.add(fn.params[0], fn.params[1])
    b.ret(s)
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    p0, p1 = fn.params[0].id, fn.params[1].id
    assert f"define i32 @addp(i32 %v{p0}, i32 %v{p1}) {{" in ll, ll
    assert f"%v{s.id} = add i32 %v{p0}, %v{p1}" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_emit_void_return():
    """A unit-returning function emits `define void` + `ret void`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("p", [], tir.TIRUnit())
    b.ret()  # RETURN with no operand
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    assert "define void @p() {" in ll, ll
    assert "ret void" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_emit_multi_function_module():
    """A module with two functions emits exactly one target-triple line
    and two `define`s, in insertion order (deterministic)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("first", [], _i32())
    b.ret(b.const_int(1))
    b.end_function()
    b.begin_function("second", [], _i32())
    b.ret(b.const_int(2))
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    assert ll.count("target triple =") == 1, ll
    assert ll.count("\ndefine ") == 2, ll
    # Insertion order preserved: `first` before `second`.
    assert ll.index("@first(") < ll.index("@second("), ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_emit_is_deterministic():
    """Two emits of the same module are byte-identical (a Stage 207
    parity-harness prerequisite)."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("main", [], _i32())
        b.ret(b.add(b.const_int(7), b.const_int(8)))
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage200_emit_non_i32_integer_widths():
    """The scalar core emits every supported integer width — i64 and
    bool->i1, not only i32."""
    for dtype, llvm_ty in (("i64", "i64"), ("bool", "i1")):
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("w", [], tir.TIRScalar(dtype))
        b.ret(b.const_int(1, dtype=dtype))
        b.end_function()
        ll = llvm_ir.emit_module(mod)
        assert f"define {llvm_ty} @w() {{" in ll, (dtype, ll)
        assert f"ret {llvm_ty} 1" in ll, (dtype, ll)
        assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_quotes_out_of_grammar_function_name():
    """A function name outside LLVM's bare-identifier grammar (here a
    monomorphized-style `foo<i32>`) is emitted in quoted `@"..."` form
    so the IR stays valid — never interpolated raw."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("foo<i32>", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert 'define i32 @"foo<i32>"() {' in ll, ll
    assert "@foo<i32>" not in ll, ll  # never the raw unquoted form
    assert llvm_ir.mock_validate_ll(ll) == []


# --------------------------------------------------------------------------
# scalar core — loud rejection of out-of-scope constructs
# --------------------------------------------------------------------------
def test_stage200_rejects_unsupported_op():
    """An op outside the scalar core (here DIV) is rejected with
    LLVMEmitError naming the op — never silently dropped."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.DIV, b.const_int(8), b.const_int(2),
               result_ty=_i32())
    b.ret(r)
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="elem.div"):
        llvm_ir.emit_module(mod)


def test_stage200_rejects_multi_block_function():
    """A function with more than one block is rejected — control flow
    is Stage 202, not Stage 200."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.append_block()  # second block -> out of scalar-core scope
    b.ret(b.const_int(0))
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="single-block"):
        llvm_ir.emit_module(mod)


def test_stage200_rejects_non_integer_return_type():
    """A non-integer return type (f32) is rejected — floats are a
    later stage."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("f32"))
    b.ret(b.const_float(1.0))
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="f32"):
        llvm_ir.emit_module(mod)


def test_stage200_rejects_missing_terminator():
    """A function whose block has no RETURN is rejected — every LLVM
    basic block needs a terminator."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("noret", [], _i32())
    b.const_int(1)  # a value, but no RETURN
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="no RETURN"):
        llvm_ir.emit_module(mod)


def test_stage200_rejects_value_used_before_definition():
    """Referencing a value with no prior definition (e.g. a foreign
    Value id) raises rather than emitting a dangling `%vN`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    stray = tir.Value(id=9999, ty=_i32())  # never defined in this fn
    b.ret(stray)
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="before it is defined"):
        llvm_ir.emit_module(mod)


def test_stage200_rejects_binop_operand_type_mismatch():
    """A binary op whose operand types differ from its result type is
    rejected — LLVM requires operand and result types to match, and
    emitting `add i32` over i64 registers would be malformed IR."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    a = b.const_int(2, dtype="i32")
    c = b.const_int(3, dtype="i32")
    # ADD with i32 operands but an i64 result.
    bad = b.emit(tir.OpKind.ADD, a, c, result_ty=tir.TIRScalar("i64"))
    b.ret(bad)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="matching operand and result types"):
        llvm_ir.emit_module(mod)


# --------------------------------------------------------------------------
# mock validation
# --------------------------------------------------------------------------
def test_stage200_mock_validate_flags_missing_terminator():
    """mock_validate_ll catches a `define` body with no `ret`."""
    bad = (
        'target triple = "x86_64-unknown-linux-gnu"\n'
        "\n"
        "define i32 @main() {\n"
        "  %v0 = add i32 1, 2\n"
        "}\n"
    )
    problems = llvm_ir.mock_validate_ll(bad)
    assert any("no `ret` terminator" in p for p in problems), problems


def test_stage200_mock_validate_flags_unbalanced_braces():
    """mock_validate_ll catches an unbalanced-brace module."""
    bad = (
        'target triple = "x86_64-unknown-linux-gnu"\n'
        "define i32 @main() {\n"
        "  ret i32 0\n"
    )  # missing closing brace
    problems = llvm_ir.mock_validate_ll(bad)
    assert any("unbalanced braces" in p for p in problems), problems


def test_stage200_mock_validate_flags_missing_triple_and_defines():
    """mock_validate_ll catches a module with no triple and no
    functions."""
    problems = llvm_ir.mock_validate_ll("; just a comment\n")
    assert any("target triple" in p for p in problems), problems
    assert any("no `define`" in p for p in problems), problems


def test_stage200_mock_validate_handles_indented_module():
    """mock_validate_ll matches line-leading tokens after stripping
    indentation, so an indented but otherwise-valid `.ll` validates
    clean (it is not coupled to column-0 emission)."""
    indented = (
        '  target triple = "x86_64-unknown-linux-gnu"\n'
        "  define i32 @main() {\n"
        "    ret i32 0\n"
        "  }\n"
    )
    assert llvm_ir.mock_validate_ll(indented) == []


def test_stage200_mock_validate_not_fooled_by_brace_in_quoted_name():
    """A `}` legally inside a quoted `@"..."` identifier must not be
    counted as a structural brace. A genuinely unbalanced module whose
    only `}` sits inside a quoted name is still flagged."""
    broken = (
        'target triple = "x86_64-unknown-linux-gnu"\n'
        'define i32 @"bad}"() {\n'
        "  ret i32 0\n"
    )  # the define body's closing `}` is missing
    problems = llvm_ir.mock_validate_ll(broken)
    assert any("unbalanced braces" in p for p in problems), problems
    # A *valid* module with a brace-bearing quoted name still validates
    # clean — the quoted-span masking does not break the happy path.
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("ok}name", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert '@"ok}name"' in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage200_rejects_const_int_bool_value():
    """A CONST_INT whose `value` attr is a Python bool is rejected —
    `isinstance(True, int)` is True, so the guard uses `type(...) is
    int` to avoid emitting a malformed `ret i32 True`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    cv = b.emit(tir.OpKind.CONST_INT, result_ty=_i32(),
                attrs={"value": True})
    b.ret(cv)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="integer 'value' attr"):
        llvm_ir.emit_module(mod)
