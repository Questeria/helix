"""Tests for helixc.backend.llvm_ir — v3.0 Phase D (Stages 200-205):
the additive LLVM IR emitter.

The emitter consumes the same host IR (`tir.Module`) that
`x86_64.py::compile_module_to_elf` consumes. Covered so far: the
scalar core (module header / target triple, `define`, integer
constants, add/sub/mul, `ret`); control flow (multi-block, `br`,
`phi`); the scalar op set (the six comparisons, `select`, `neg`,
div/mod, bitwise ops, unsigned integer dtypes); mutable local
variables and stack arrays (`alloca`/`load`/`store`/`getelementptr`);
and direct + FFI function calls (`call`, with a module-scope
`declare` for FFI targets). Everything else must be REJECTED loudly
with `LLVMEmitError` — a partial backend fails closed, it never emits
wrong IR.
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
    """An op outside the supported set (here MAXIMUM) is rejected with
    LLVMEmitError naming the op — never silently dropped."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.MAXIMUM, b.const_int(8), b.const_int(2),
               result_ty=_i32())
    b.ret(r)
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="elem.maximum"):
        llvm_ir.emit_module(mod)


def test_stage202_emit_straight_multiblock():
    """Stage 202 — a two-block function (entry unconditionally branches
    to a second block which returns) emits both blocks with labels and
    a `br`. Multi-block is now supported, not rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    second = b.append_block()
    b.emit(tir.OpKind.BR, attrs={"target_block": second.id})
    b.switch_to(second)
    b.ret(b.const_int(7))
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    assert f"br label %bb{second.id}" in ll, ll
    assert "ret i32 7" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


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

    with pytest.raises(llvm_ir.LLVMEmitError, match="no terminator"):
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

    with pytest.raises(llvm_ir.LLVMEmitError, match="defined by no op"):
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
    """mock_validate_ll catches a `define` body whose last instruction
    is not a basic-block terminator."""
    bad = (
        'target triple = "x86_64-unknown-linux-gnu"\n'
        "\n"
        "define i32 @main() {\n"
        "  %v0 = add i32 1, 2\n"
        "}\n"
    )
    problems = llvm_ir.mock_validate_ll(bad)
    assert any("does not end with a terminator" in p
               for p in problems), problems


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


# ==========================================================================
# Stage 202 — control flow (multi-block, br, phi)
# ==========================================================================
def test_stage202_emit_if_else_with_phi():
    """An if/else: the entry COND_BRs to two arms, each BRs to a merge
    block whose parameter becomes a `phi` collecting both arms'
    values."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("pick", [("c", tir.TIRScalar("bool"))], _i32())
    then_blk = b.append_block()
    else_blk = b.append_block()
    merge = b.append_block()
    b.emit(tir.OpKind.COND_BR, fn.params[0],
           attrs={"true_block": then_blk.id, "false_block": else_blk.id})
    b.switch_to(then_blk)
    b.emit(tir.OpKind.BR, b.const_int(10),
           attrs={"target_block": merge.id})
    b.switch_to(else_blk)
    b.emit(tir.OpKind.BR, b.const_int(20),
           attrs={"target_block": merge.id})
    b.switch_to(merge)
    p = b.new_block_param(_i32())
    b.ret(p)
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    assert (f"br i1 %v{fn.params[0].id}, label %bb{then_blk.id}, "
            f"label %bb{else_blk.id}") in ll, ll
    assert (f"%v{p.id} = phi i32 [ 10, %bb{then_blk.id} ], "
            f"[ 20, %bb{else_blk.id} ]") in ll, ll
    assert f"ret i32 %v{p.id}" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage202_emit_loop_phi_forward_reference():
    """A loop header's `phi` references a value defined LATER on the
    back-edge — the pre-pass registers every value up front so the
    forward reference resolves (LLVM textual IR permits it)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "loop", [("start", _i32()), ("again", tir.TIRScalar("bool"))],
        _i32())
    header = b.append_block()
    body = b.append_block()
    exit_blk = b.append_block()
    b.emit(tir.OpKind.BR, fn.params[0],
           attrs={"target_block": header.id})
    b.switch_to(header)
    acc = b.new_block_param(_i32())
    b.emit(tir.OpKind.COND_BR, fn.params[1],
           attrs={"true_block": body.id, "false_block": exit_blk.id})
    b.switch_to(body)
    acc2 = b.add(acc, acc)
    b.emit(tir.OpKind.BR, acc2, attrs={"target_block": header.id})
    b.switch_to(exit_blk)
    b.ret(acc)
    b.end_function()

    ll = llvm_ir.emit_module(mod)
    entry_id = fn.blocks[0].id
    assert f"%v{acc.id} = phi i32" in ll, ll
    # entry edge + the body back-edge; acc2 is defined later, in `body`.
    assert f"[ %v{fn.params[0].id}, %bb{entry_id} ]" in ll, ll
    assert f"[ %v{acc2.id}, %bb{body.id} ]" in ll, ll
    assert f"%v{acc2.id} = add i32 %v{acc.id}, %v{acc.id}" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage202_rejects_branch_to_entry_block():
    """LLVM forbids branching to a function's entry block — a BR that
    targets block 0 is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [], _i32())
    entry_id = fn.blocks[0].id
    second = b.append_block()
    b.emit(tir.OpKind.BR, attrs={"target_block": second.id})
    b.switch_to(second)
    b.emit(tir.OpKind.BR, attrs={"target_block": entry_id})  # -> entry
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="entry block"):
        llvm_ir.emit_module(mod)


def test_stage202_rejects_non_i1_cond_br_condition():
    """A COND_BR whose condition is not i1 (here an i32 parameter) is
    rejected — LLVM `br` requires an i1 condition."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    t_blk = b.append_block()
    f_blk = b.append_block()
    b.emit(tir.OpKind.COND_BR, fn.params[0],
           attrs={"true_block": t_blk.id, "false_block": f_blk.id})
    b.switch_to(t_blk)
    b.ret(b.const_int(1))
    b.switch_to(f_blk)
    b.ret(b.const_int(0))
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError, match="i1"):
        llvm_ir.emit_module(mod)


def test_stage202_rejects_phi_input_from_cond_br():
    """A block with parameters reached via a COND_BR is rejected — a
    COND_BR carries no branch arguments, so it cannot supply phi
    inputs."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("c", tir.TIRScalar("bool"))], _i32())
    target = b.append_block()
    other = b.append_block()
    b.emit(tir.OpKind.COND_BR, fn.params[0],
           attrs={"true_block": target.id, "false_block": other.id})
    b.switch_to(target)
    p = b.new_block_param(_i32())   # a param, but COND_BR supplies no arg
    b.ret(p)
    b.switch_to(other)
    b.ret(b.const_int(0))
    b.end_function()

    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="only BR can supply phi inputs"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 203 — scalar op set (comparisons, select, neg, unsigned dtypes)
# ==========================================================================
def test_stage203_emit_signed_comparison():
    """A comparison of signed integers uses a signed icmp predicate."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("cmp", [("a", _i32()), ("c", _i32())],
                          tir.TIRScalar("bool"))
    r = b.emit(tir.OpKind.CMP_LT, fn.params[0], fn.params[1],
               result_ty=tir.TIRScalar("bool"))
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = icmp slt i32 %v{fn.params[0].id}, "
            f"%v{fn.params[1].id}") in ll, ll
    assert f"ret i1 %v{r.id}" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_emit_unsigned_comparison():
    """A comparison of unsigned integers uses an unsigned icmp
    predicate — the signedness follows the operand dtype."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("ucmp", [("a", u32), ("c", u32)],
                          tir.TIRScalar("bool"))
    r = b.emit(tir.OpKind.CMP_LT, fn.params[0], fn.params[1],
               result_ty=tir.TIRScalar("bool"))
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = icmp ult i32" in ll, ll
    assert "icmp slt" not in ll, ll  # NOT the signed predicate


def test_stage203_emit_eq_comparison_is_sign_agnostic():
    """`==` lowers to `icmp eq` regardless of operand signedness."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("eq", [("a", u32), ("c", u32)],
                          tir.TIRScalar("bool"))
    r = b.emit(tir.OpKind.CMP_EQ, fn.params[0], fn.params[1],
               result_ty=tir.TIRScalar("bool"))
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = icmp eq i32" in ll, ll


def test_stage203_emit_select():
    """SELECT lowers to LLVM `select i1`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("sel", [("c", tir.TIRScalar("bool"))], _i32())
    r = b.emit(tir.OpKind.SELECT, fn.params[0], b.const_int(10),
               b.const_int(20), result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = select i1 %v{fn.params[0].id}, "
            f"i32 10, i32 20") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_emit_neg():
    """NEG lowers to `sub <ty> 0, x` — LLVM has no integer negate."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("neg", [("x", _i32())], _i32())
    r = b.emit(tir.OpKind.NEG, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = sub i32 0, %v{fn.params[0].id}" in ll, ll


def test_stage203_emit_unsigned_arithmetic():
    """An unsigned dtype shares its signed counterpart's LLVM integer
    type — `add` on u32 emits `add i32` (LLVM `add` is sign-agnostic)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("uadd", [("a", u32), ("c", u32)], u32)
    r = b.emit(tir.OpKind.ADD, fn.params[0], fn.params[1], result_ty=u32)
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "define i32 @uadd(i32 %v0, i32 %v1) {" in ll, ll
    assert f"%v{r.id} = add i32 %v0, %v1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_comparison_feeds_cond_br():
    """A comparison's i1 result is a valid COND_BR condition — the
    realistic if-pattern (`icmp` then `br i1`)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    t_blk = b.append_block()
    f_blk = b.append_block()
    cmp = b.emit(tir.OpKind.CMP_GT, fn.params[0], b.const_int(0),
                 result_ty=tir.TIRScalar("bool"))
    b.emit(tir.OpKind.COND_BR, cmp,
           attrs={"true_block": t_blk.id, "false_block": f_blk.id})
    b.switch_to(t_blk)
    b.ret(b.const_int(1))
    b.switch_to(f_blk)
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{cmp.id} = icmp sgt i32 %v{fn.params[0].id}, 0" in ll, ll
    assert (f"br i1 %v{cmp.id}, label %bb{t_blk.id}, "
            f"label %bb{f_blk.id}") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_rejects_non_bool_comparison_result():
    """A comparison whose result type is not bool/i1 is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.CMP_EQ, fn.params[0], fn.params[1],
               result_ty=_i32())  # i32 result, not bool
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="produces a bool"):
        llvm_ir.emit_module(mod)


def test_stage203_rejects_non_i1_select_condition():
    """A SELECT whose condition is not i1 is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    r = b.emit(tir.OpKind.SELECT, fn.params[0], b.const_int(1),
               b.const_int(2), result_ty=_i32())  # i32 condition
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="i1"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 202+203 audit-fix regression tests
# ==========================================================================
def test_stage202_rejects_phi_arg_type_mismatch():
    """A BR whose argument type differs from the target block's
    parameter type is rejected — a phi incoming must match the
    parameter type (else the emitted phi references a wrong-width
    register)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    merge = b.append_block()
    # entry: br [an i64 constant] -> merge, whose parameter is i32.
    b.emit(tir.OpKind.BR, b.const_int(5, dtype="i64"),
           attrs={"target_block": merge.id})
    b.switch_to(merge)
    b.new_block_param(_i32())   # i32 param, but the BR arg is i64
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="must match the parameter type"):
        llvm_ir.emit_module(mod)


def test_stage202_mock_validate_accepts_retless_infinite_loop():
    """A valid multi-block function with no `ret` at all — an infinite
    loop where every block ends in `br` — passes mock validation. The
    check is 'the body ends with a terminator', not 'has a ret'."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("spin", [], _i32())
    loop = b.append_block()
    b.emit(tir.OpKind.BR, attrs={"target_block": loop.id})
    b.switch_to(loop)
    b.emit(tir.OpKind.BR, attrs={"target_block": loop.id})  # br to self
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "ret " not in ll, ll  # genuinely no `ret` instruction
    assert llvm_ir.mock_validate_ll(ll) == [], llvm_ir.mock_validate_ll(ll)


# ==========================================================================
# Stage 203 (cont.) — division / remainder, bitwise ops, shifts
# ==========================================================================
def test_stage203_emit_signed_division():
    """DIV on signed integers lowers to LLVM `sdiv`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("d", [("a", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.DIV, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = sdiv i32 %v{fn.params[0].id}, "
            f"%v{fn.params[1].id}") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_emit_unsigned_division():
    """DIV on unsigned integers lowers to `udiv` — the signedness
    follows the operand dtype, never the signed `sdiv`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("ud", [("a", u32), ("c", u32)], u32)
    r = b.emit(tir.OpKind.DIV, fn.params[0], fn.params[1], result_ty=u32)
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = udiv i32" in ll, ll
    assert "sdiv" not in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_emit_signed_remainder():
    """MOD on signed integers lowers to LLVM `srem`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("m", [("a", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.MOD, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = srem i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_emit_unsigned_remainder():
    """MOD on unsigned integers lowers to `urem`, never the signed
    `srem`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("um", [("a", u32), ("c", u32)], u32)
    r = b.emit(tir.OpKind.MOD, fn.params[0], fn.params[1], result_ty=u32)
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = urem i32" in ll, ll
    assert "srem" not in ll, ll


def test_stage203_emit_bitwise_and_or_xor():
    """BIT_AND / BIT_OR / BIT_XOR lower to the sign-agnostic LLVM
    `and` / `or` / `xor`."""
    for kind, mnemonic in ((tir.OpKind.BIT_AND, "and"),
                           (tir.OpKind.BIT_OR, "or"),
                           (tir.OpKind.BIT_XOR, "xor")):
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        fn = b.begin_function(
            "bw", [("a", _i32()), ("c", _i32())], _i32())
        r = b.emit(kind, fn.params[0], fn.params[1], result_ty=_i32())
        b.ret(r)
        b.end_function()
        ll = llvm_ir.emit_module(mod)
        assert (f"%v{r.id} = {mnemonic} i32 %v{fn.params[0].id}, "
                f"%v{fn.params[1].id}") in ll, (mnemonic, ll)
        assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_bitwise_is_sign_agnostic():
    """A bitwise op on unsigned operands emits the same mnemonic as on
    signed ones — `and`/`or`/`xor` have no signed/unsigned variants."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("uand", [("a", u32), ("c", u32)], u32)
    r = b.emit(tir.OpKind.BIT_AND, fn.params[0], fn.params[1],
               result_ty=u32)
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = and i32" in ll, ll


def test_stage203_emit_shift_left():
    """SHL lowers to LLVM `shl` (sign-agnostic — it stays in the
    sign-agnostic binop table)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("sl", [("x", _i32()), ("n", _i32())], _i32())
    r = b.emit(tir.OpKind.SHL, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = shl i32 %v{fn.params[0].id}, "
            f"%v{fn.params[1].id}") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_emit_signed_shift_right():
    """SHR on a signed value lowers to the arithmetic `ashr` — the
    vacated high bits take the sign bit."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("sr", [("x", _i32()), ("n", _i32())], _i32())
    r = b.emit(tir.OpKind.SHR, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = ashr i32" in ll, ll
    assert "lshr" not in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_emit_unsigned_shift_right():
    """SHR on an unsigned value lowers to the logical `lshr` — the
    vacated high bits are zero-filled. The signedness follows the
    shifted value's dtype."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("usr", [("x", u32), ("n", u32)], u32)
    r = b.emit(tir.OpKind.SHR, fn.params[0], fn.params[1], result_ty=u32)
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = lshr i32" in ll, ll
    assert "ashr" not in ll, ll


def test_stage203_emit_bitwise_not():
    """BIT_NOT lowers to `xor <ty> x, -1` — LLVM has no bitwise-NOT
    instruction; xor against all-ones flips every bit."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("bn", [("x", _i32())], _i32())
    r = b.emit(tir.OpKind.BIT_NOT, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = xor i32 %v{fn.params[0].id}, -1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_division_operand_type_mismatch_rejected():
    """DIV shares the binop type guard — operands and result must share
    one LLVM type, else the emit raises rather than producing malformed
    IR (an `sdiv i64` over i32 registers)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    a = b.const_int(8, dtype="i32")
    c = b.const_int(2, dtype="i32")
    bad = b.emit(tir.OpKind.DIV, a, c, result_ty=tir.TIRScalar("i64"))
    b.ret(bad)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="matching operand and result types"):
        llvm_ir.emit_module(mod)


def test_stage203_bit_not_operand_type_mismatch_rejected():
    """BIT_NOT's operand and result must share one type — the unified
    unary branch's type guard covers the new op too."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    x = b.const_int(5, dtype="i64")
    bad = b.emit(tir.OpKind.BIT_NOT, x, result_ty=_i32())
    b.ret(bad)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="must share one type"):
        llvm_ir.emit_module(mod)


def test_stage203_div_then_mod_share_a_block():
    """A realistic mixed expression — `(a / b) % c` — emits an `sdiv`
    feeding an `srem` in one straight-line block."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "divmod", [("a", _i32()), ("c", _i32()), ("d", _i32())], _i32())
    q = b.emit(tir.OpKind.DIV, fn.params[0], fn.params[1],
               result_ty=_i32())
    r = b.emit(tir.OpKind.MOD, q, fn.params[2], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{q.id} = sdiv i32" in ll, ll
    assert f"%v{r.id} = srem i32 %v{q.id}, %v{fn.params[2].id}" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


# --------------------------------------------------------------------------
# Stage 203 (cont.) audit-fix — mixed-sign operands fail closed for the
# ops whose LLVM instruction is CHOSEN BY signedness; sign-agnostic ops
# and shifts stay permissive.
# --------------------------------------------------------------------------
def test_stage203_rejects_mixed_sign_division():
    """A DIV whose operands disagree on signedness (i32 / u32) is
    rejected — `sdiv` vs `udiv` is ambiguous, so the backend fails
    closed rather than silently picking operand 0's interpretation."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "dm", [("a", _i32()), ("c", tir.TIRScalar("u32"))], _i32())
    r = b.emit(tir.OpKind.DIV, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="disagree on signedness"):
        llvm_ir.emit_module(mod)


def test_stage203_rejects_mixed_sign_remainder():
    """A MOD whose operands disagree on signedness is rejected for the
    same reason — `srem` vs `urem` is ambiguous."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "mm", [("a", tir.TIRScalar("u32")), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.MOD, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="disagree on signedness"):
        llvm_ir.emit_module(mod)


def test_stage203_rejects_mixed_sign_ordered_comparison():
    """An ordered comparison (`<`) whose operands disagree on
    signedness is rejected — the signed/unsigned `icmp` predicate is
    ambiguous. (`==`/`!=` stay sign-agnostic — see the next test.)"""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "cm", [("a", _i32()), ("c", tir.TIRScalar("u32"))],
        tir.TIRScalar("bool"))
    r = b.emit(tir.OpKind.CMP_LT, fn.params[0], fn.params[1],
               result_ty=tir.TIRScalar("bool"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="disagree on signedness"):
        llvm_ir.emit_module(mod)


def test_stage203_allows_mixed_sign_equality():
    """`==` is sign-agnostic (`icmp eq`) — a mixed signed/unsigned
    operand pair is accepted, not rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "eqm", [("a", _i32()), ("c", tir.TIRScalar("u32"))],
        tir.TIRScalar("bool"))
    r = b.emit(tir.OpKind.CMP_EQ, fn.params[0], fn.params[1],
               result_ty=tir.TIRScalar("bool"))
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = icmp eq i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_allows_mixed_sign_bitwise():
    """The bitwise ops are sign-agnostic — a mixed signed/unsigned
    operand pair emits the same `and` and is accepted."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "bwm", [("a", _i32()), ("c", tir.TIRScalar("u32"))], _i32())
    r = b.emit(tir.OpKind.BIT_AND, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = and i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_allows_mixed_sign_shift():
    """A shift may mix operand signedness — the shift COUNT's sign
    never affects the result, so a signed value shifted by an
    unsigned-typed amount is well-defined (not rejected). The
    arithmetic-vs-logical choice still follows the shifted VALUE
    (operand 0): a signed value -> `ashr`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "shm", [("x", _i32()), ("n", tir.TIRScalar("u32"))], _i32())
    r = b.emit(tir.OpKind.SHR, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = ashr i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage203_rejects_shift_value_result_signedness_mismatch():
    """A SHR whose shifted value and result disagree on signedness is
    rejected. `ashr` vs `lshr` is chosen by the value's sign, so a
    value/result signedness mismatch is ill-specified; failing closed
    keeps the LLVM backend's shift choice provably equal to
    x86_64.py's (which keys the choice off the result type) for every
    SHR the LLVM backend accepts."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    u32 = tir.TIRScalar("u32")
    fn = b.begin_function("svr", [("x", u32), ("n", u32)], _i32())
    r = b.emit(tir.OpKind.SHR, fn.params[0], fn.params[1],
               result_ty=_i32())  # u32 value shifted, i32 result
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="disagree on signedness"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 204 — mutable local variables (alloca / load / store)
# ==========================================================================
def test_stage204_emit_alloc_store_load():
    """A mutable local: ALLOC_VAR -> `alloca`, STORE_VAR -> `store`,
    LOAD_VAR -> `load` (opaque-pointer form)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.emit(tir.OpKind.STORE_VAR, b.const_int(42), attrs={"name": "x"})
    ld = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                attrs={"name": "x"})
    b.ret(ld)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%slot.0 = alloca i32" in ll, ll
    assert "store i32 42, ptr %slot.0" in ll, ll
    assert f"%v{ld.id} = load i32, ptr %slot.0" in ll, ll
    assert f"ret i32 %v{ld.id}" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_alloca_hoisted_to_entry_block():
    """An ALLOC_VAR that appears textually in a non-entry block still
    has its `alloca` emitted in the entry block — the LLVM convention,
    and the entry block dominates every use."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [], _i32())
    second = b.append_block()
    b.emit(tir.OpKind.BR, attrs={"target_block": second.id})
    b.switch_to(second)
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.emit(tir.OpKind.STORE_VAR, b.const_int(7), attrs={"name": "x"})
    ld = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                attrs={"name": "x"})
    b.ret(ld)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    entry_id = fn.blocks[0].id
    entry_seg = ll[ll.index(f"bb{entry_id}:"):ll.index(f"bb{second.id}:")]
    assert "%slot.0 = alloca i32" in entry_seg, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_load_store_in_non_entry_block():
    """A LOAD_VAR / STORE_VAR in a non-entry block resolves the slot
    allocated (and hoisted) in the entry block — the entry alloca
    dominates every block."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    second = b.append_block()
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.emit(tir.OpKind.BR, attrs={"target_block": second.id})
    b.switch_to(second)
    b.emit(tir.OpKind.STORE_VAR, b.const_int(9), attrs={"name": "x"})
    ld = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                attrs={"name": "x"})
    b.ret(ld)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%slot.0 = alloca i32" in ll, ll
    assert "store i32 9, ptr %slot.0" in ll, ll
    assert f"%v{ld.id} = load i32, ptr %slot.0" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_multiple_vars_distinct_slots():
    """Two mutable locals get two allocas with distinct counter-named
    registers (%slot.0, %slot.1)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("two", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "a", "dtype": _i32()})
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "c", "dtype": _i32()})
    b.emit(tir.OpKind.STORE_VAR, b.const_int(1), attrs={"name": "a"})
    b.emit(tir.OpKind.STORE_VAR, b.const_int(2), attrs={"name": "c"})
    la = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                attrs={"name": "a"})
    b.ret(la)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%slot.0 = alloca i32" in ll, ll
    assert "%slot.1 = alloca i32" in ll, ll
    assert "store i32 1, ptr %slot.0" in ll, ll
    assert "store i32 2, ptr %slot.1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_round_trip_mutable_counter():
    """A realistic mutable local — `let mut x = 5; x = x + 1; x` —
    lowers to alloca + store + (load, add, store) + load."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("count", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.emit(tir.OpKind.STORE_VAR, b.const_int(5), attrs={"name": "x"})
    cur = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                 attrs={"name": "x"})
    inc = b.add(cur, b.const_int(1))
    b.emit(tir.OpKind.STORE_VAR, inc, attrs={"name": "x"})
    final = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                   attrs={"name": "x"})
    b.ret(final)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%slot.0 = alloca i32" in ll, ll
    assert "store i32 5, ptr %slot.0" in ll, ll
    assert f"%v{cur.id} = load i32, ptr %slot.0" in ll, ll
    assert f"%v{inc.id} = add i32 %v{cur.id}, 1" in ll, ll
    assert f"store i32 %v{inc.id}, ptr %slot.0" in ll, ll
    assert f"%v{final.id} = load i32, ptr %slot.0" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_emit_is_deterministic_with_allocas():
    """Two emits of a module with mutable locals are byte-identical —
    the alloca order (ALLOC_VAR-encounter order) is deterministic."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        b.emit(tir.OpKind.ALLOC_VAR,
               attrs={"name": "a", "dtype": _i32()})
        b.emit(tir.OpKind.ALLOC_VAR,
               attrs={"name": "c", "dtype": _i32()})
        b.emit(tir.OpKind.STORE_VAR, b.const_int(1),
               attrs={"name": "a"})
        ld = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                    attrs={"name": "a"})
        b.ret(ld)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage204_rejects_load_undeclared_var():
    """A LOAD_VAR naming a variable that no ALLOC_VAR declares is
    rejected — never a dangling load of an unknown slot."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    ld = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                attrs={"name": "ghost"})
    b.ret(ld)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="no ALLOC_VAR"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_store_undeclared_var():
    """A STORE_VAR naming a variable that no ALLOC_VAR declares is
    rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.STORE_VAR, b.const_int(1),
           attrs={"name": "ghost"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="no ALLOC_VAR"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_duplicate_alloc_var():
    """Two ALLOC_VAR ops with the same name are rejected — lower_ast
    mangles shadowed locals to unique names, so a duplicate name is
    malformed IR."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="declared more than once"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_alloc_var_with_result():
    """An ALLOC_VAR carrying a result is rejected — the op declares a
    cell, it produces no SSA value."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, result_ty=_i32(),
           attrs={"name": "x", "dtype": _i32()})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ALLOC_VAR produces no result"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_load_var_type_mismatch():
    """A LOAD_VAR whose result type differs from the cell's allocated
    type is rejected — a load must read the type the cell holds."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    ld = b.emit(tir.OpKind.LOAD_VAR, result_ty=tir.TIRScalar("i64"),
                attrs={"name": "x"})
    b.ret(ld)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="a load must read the type"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_store_var_type_mismatch():
    """A STORE_VAR whose value type differs from the cell's allocated
    type is rejected — a store must write the type the cell holds."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.emit(tir.OpKind.STORE_VAR, b.const_int(1, dtype="i64"),
           attrs={"name": "x"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="a store must write the type"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_non_scalar_alloc_var():
    """An ALLOC_VAR whose dtype is not a scalar integer (here f32) is
    rejected — Stage 204 allocates only scalar integer cells."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR,
           attrs={"name": "x", "dtype": tir.TIRScalar("f32")})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="f32"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 204 sub-stage B — stack arrays (alloca [N x T] / getelementptr)
# ==========================================================================
def test_stage204_emit_alloc_array_store_load_elem():
    """A stack array: ALLOC_ARRAY -> array-typed `alloca`, STORE_ELEM /
    LOAD_ELEM -> `getelementptr` + `store` / `load`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 4})
    b.emit(tir.OpKind.STORE_ELEM, b.const_int(0), b.const_int(42),
           attrs={"name": "xs"})
    ld = b.emit(tir.OpKind.LOAD_ELEM, b.const_int(0), result_ty=_i32(),
                attrs={"name": "xs"})
    b.ret(ld)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%arr.0 = alloca [4 x i32]" in ll, ll
    assert ("%gep.0 = getelementptr [4 x i32], ptr %arr.0, "
            "i64 0, i32 0") in ll, ll
    assert "store i32 42, ptr %gep.0" in ll, ll
    assert "%gep.1 = getelementptr [4 x i32], ptr %arr.0, i64 0" in ll, ll
    assert f"%v{ld.id} = load i32, ptr %gep.1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_array_alloca_hoisted_to_entry():
    """An ALLOC_ARRAY in a non-entry block still has its array-typed
    `alloca` emitted in the entry block; LOAD_ELEM / STORE_ELEM work
    from a non-entry block too."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [], _i32())
    second = b.append_block()
    b.emit(tir.OpKind.BR, attrs={"target_block": second.id})
    b.switch_to(second)
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 3})
    b.emit(tir.OpKind.STORE_ELEM, b.const_int(0), b.const_int(5),
           attrs={"name": "xs"})
    ld = b.emit(tir.OpKind.LOAD_ELEM, b.const_int(0), result_ty=_i32(),
                attrs={"name": "xs"})
    b.ret(ld)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    entry_id = fn.blocks[0].id
    entry_seg = ll[ll.index(f"bb{entry_id}:"):ll.index(f"bb{second.id}:")]
    assert "%arr.0 = alloca [3 x i32]" in entry_seg, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_load_elem_runtime_index():
    """LOAD_ELEM with a runtime (non-constant) index emits a GEP whose
    element index is the SSA register of the index value."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], _i32())
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 8})
    ld = b.emit(tir.OpKind.LOAD_ELEM, fn.params[0], result_ty=_i32(),
                attrs={"name": "xs"})
    b.ret(ld)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"getelementptr [8 x i32], ptr %arr.0, i64 0, "
            f"i32 %v{fn.params[0].id}") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_array_and_scalar_var_coexist():
    """A function with both a mutable local and a stack array gets a
    %slot.N alloca and a %arr.N alloca — distinct register namespaces."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "n", "dtype": _i32()})
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 2})
    b.emit(tir.OpKind.STORE_VAR, b.const_int(1), attrs={"name": "n"})
    ld = b.emit(tir.OpKind.LOAD_VAR, result_ty=_i32(),
                attrs={"name": "n"})
    b.ret(ld)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%slot.0 = alloca i32" in ll, ll
    assert "%arr.0 = alloca [2 x i32]" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_multiple_arrays_distinct_slots():
    """Two stack arrays get two allocas with distinct counter-named
    registers (%arr.0, %arr.1) and their own element types."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "a", "dtype": _i32(), "length": 2})
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "c", "dtype": tir.TIRScalar("i64"),
                  "length": 3})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%arr.0 = alloca [2 x i32]" in ll, ll
    assert "%arr.1 = alloca [3 x i64]" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage204_array_emit_is_deterministic():
    """Two emits of a module with a stack array are byte-identical —
    the alloca and %gep.N numbering are deterministic."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        b.emit(tir.OpKind.ALLOC_ARRAY,
               attrs={"name": "xs", "dtype": _i32(), "length": 3})
        b.emit(tir.OpKind.STORE_ELEM, b.const_int(0), b.const_int(7),
               attrs={"name": "xs"})
        ld = b.emit(tir.OpKind.LOAD_ELEM, b.const_int(1),
                    result_ty=_i32(), attrs={"name": "xs"})
        b.ret(ld)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage204_rejects_load_elem_undeclared_array():
    """A LOAD_ELEM naming an array that no ALLOC_ARRAY declares is
    rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    ld = b.emit(tir.OpKind.LOAD_ELEM, b.const_int(0), result_ty=_i32(),
                attrs={"name": "ghost"})
    b.ret(ld)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="no ALLOC_ARRAY"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_store_elem_undeclared_array():
    """A STORE_ELEM naming an array that no ALLOC_ARRAY declares is
    rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.STORE_ELEM, b.const_int(0), b.const_int(1),
           attrs={"name": "ghost"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="no ALLOC_ARRAY"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_load_elem_type_mismatch():
    """A LOAD_ELEM whose result type differs from the array's element
    type is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 4})
    ld = b.emit(tir.OpKind.LOAD_ELEM, b.const_int(0),
                result_ty=tir.TIRScalar("i64"), attrs={"name": "xs"})
    b.ret(ld)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="must read the element type"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_store_elem_type_mismatch():
    """A STORE_ELEM whose value type differs from the array's element
    type is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 4})
    b.emit(tir.OpKind.STORE_ELEM, b.const_int(0),
           b.const_int(1, dtype="i64"), attrs={"name": "xs"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="must write the element type"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_duplicate_alloc_array():
    """Two ALLOC_ARRAY ops with the same name are rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 2})
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 2})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="declared more than once"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_var_array_name_collision():
    """A name declared by both an ALLOC_VAR and an ALLOC_ARRAY is
    rejected — slot names must be unique across both tables."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_VAR, attrs={"name": "x", "dtype": _i32()})
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "x", "dtype": _i32(), "length": 2})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="declared more than once"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_alloc_array_zero_length():
    """An ALLOC_ARRAY with a non-positive length is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": _i32(), "length": 0})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="positive integer 'length'"):
        llvm_ir.emit_module(mod)


def test_stage204_rejects_non_scalar_array_element():
    """An ALLOC_ARRAY whose element dtype is not a scalar integer
    (here f32) is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.ALLOC_ARRAY,
           attrs={"name": "xs", "dtype": tir.TIRScalar("f32"),
                  "length": 3})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="f32"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 205 chunk A — direct function calls (CALL -> LLVM `call`)
# ==========================================================================
def test_stage205_emit_call_with_result():
    """A CALL with a scalar result lowers to
    `%vN = call <ty> @target(...)`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.CALL, b.const_int(7), result_ty=_i32(),
               attrs={"target": "callee"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = call i32 @callee(i32 7)" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_emit_call_void():
    """A CALL with no result lowers to `call void @target(...)`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.CALL, attrs={"target": "side_effect"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "call void @side_effect()" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_emit_call_unit_result():
    """A CALL whose single result is the unit type lowers to a void
    `call` — `()` is not a materialized LLVM value, so no register."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.CALL, result_ty=tir.TIRUnit(),
           attrs={"target": "p"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "call void @p()" in ll, ll
    assert "= call" not in ll, ll  # no result register for a unit call
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_emit_call_with_args():
    """A CALL passes each operand as a positional typed LLVM
    argument."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.CALL, fn.params[0], fn.params[1],
               result_ty=_i32(), attrs={"target": "g"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @g(i32 %v{fn.params[0].id}, "
            f"i32 %v{fn.params[1].id})") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_call_result_feeds_op():
    """A CALL's result is a normal SSA value — it can feed a later
    op."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    c = b.emit(tir.OpKind.CALL, result_ty=_i32(), attrs={"target": "g"})
    s = b.add(c, b.const_int(1))
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{c.id} = call i32 @g()" in ll, ll
    assert f"%v{s.id} = add i32 %v{c.id}, 1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_call_quotes_out_of_grammar_target():
    """A callee name outside LLVM's bare-identifier grammar (here a
    monomorphized-style name) is emitted in quoted `@"..."` form."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.CALL, result_ty=_i32(),
               attrs={"target": "gen<i32>"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert '@"gen<i32>"' in ll, ll
    assert "@gen<i32>" not in ll, ll  # never the raw unquoted form
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_emit_caller_callee_module():
    """A two-function module: one function calls the other; both
    `define`s are emitted and the call resolves by name."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("callee", [("x", _i32())], _i32())
    b.ret(b.add(b.const_int(1), b.const_int(2)))
    b.end_function()
    b.begin_function("caller", [], _i32())
    r = b.emit(tir.OpKind.CALL, b.const_int(5), result_ty=_i32(),
               attrs={"target": "callee"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "define i32 @callee(" in ll, ll
    assert "define i32 @caller(" in ll, ll
    assert f"%v{r.id} = call i32 @callee(i32 5)" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_call_emit_is_deterministic():
    """Two emits of a module with a CALL are byte-identical."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        r = b.emit(tir.OpKind.CALL, b.const_int(3), result_ty=_i32(),
                   attrs={"target": "g"})
        b.ret(r)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage205_rejects_call_missing_target():
    """A CALL with no 'target' attr is rejected — never an anonymous
    `call`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.CALL, result_ty=_i32(), attrs={})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="non-empty string 'target'"):
        llvm_ir.emit_module(mod)


def test_stage205_rejects_call_non_int_result():
    """A CALL whose result type is not a scalar integer (here f32) is
    rejected — floats are a later stage."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.CALL, result_ty=tir.TIRScalar("f32"),
           attrs={"target": "g"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="f32"):
        llvm_ir.emit_module(mod)


def test_stage205_rejects_call_multiple_results():
    """A CALL with more than one result is rejected — an LLVM call
    yields at most one value. Not constructible via `IRBuilder.emit`
    (which makes <=1 result), so the op is built directly to pin the
    fail-closed guard."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r1 = tir.Value(id=900, ty=_i32())
    r2 = tir.Value(id=901, ty=_i32())
    b.current_block.ops.append(
        tir.Op(kind=tir.OpKind.CALL, operands=[], results=[r1, r2],
               attrs={"target": "g"}))
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="at most one value"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 205 chunk B — FFI calls (FFI_CALL -> `call` + module `declare`)
# ==========================================================================
def test_stage205_emit_ffi_call_emits_declare():
    """An FFI_CALL emits the LLVM `call` plus a module-scope `declare`
    for the extern target."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.FFI_CALL, b.const_int(65), result_ty=_i32(),
               attrs={"target": "putchar"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "declare i32 @putchar(i32)" in ll, ll
    assert f"%v{r.id} = call i32 @putchar(i32 65)" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_ffi_call_void():
    """An FFI_CALL with no result emits a `declare void` and a
    `call void`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.FFI_CALL, attrs={"target": "abort"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "declare void @abort()" in ll, ll
    assert "call void @abort()" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_ffi_call_with_args():
    """An FFI_CALL's `declare` lists each argument's LLVM type."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("c", tir.TIRScalar("i64"))], _i32())
    r = b.emit(tir.OpKind.FFI_CALL, fn.params[0], fn.params[1],
               result_ty=_i32(), attrs={"target": "ext2"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "declare i32 @ext2(i32, i64)" in ll, ll
    assert (f"%v{r.id} = call i32 @ext2(i32 %v{fn.params[0].id}, "
            f"i64 %v{fn.params[1].id})") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_ffi_declare_deduped():
    """Two FFI_CALLs to the same extern symbol emit exactly one
    `declare`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.FFI_CALL, b.const_int(1), result_ty=_i32(),
           attrs={"target": "ext"})
    r = b.emit(tir.OpKind.FFI_CALL, b.const_int(2), result_ty=_i32(),
               attrs={"target": "ext"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("declare i32 @ext(i32)") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_ffi_declare_precedes_defines():
    """The FFI `declare`s are emitted at module scope, before the
    function `define`s."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.FFI_CALL, result_ty=_i32(),
               attrs={"target": "ext"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.index("declare i32 @ext()") < ll.index(
        "define i32 @f("), ll


def test_stage205_ffi_call_result_feeds_op():
    """An FFI_CALL's result is a normal SSA value — it can feed a
    later op."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    c = b.emit(tir.OpKind.FFI_CALL, result_ty=_i32(),
               attrs={"target": "ext"})
    s = b.add(c, b.const_int(10))
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{c.id} = call i32 @ext()" in ll, ll
    assert f"%v{s.id} = add i32 %v{c.id}, 10" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_ffi_and_direct_call_coexist():
    """A function may mix a direct CALL and an FFI_CALL; only the FFI
    target gets a `declare`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("helper", [], _i32())
    b.ret(b.const_int(1))
    b.end_function()
    b.begin_function("f", [], _i32())
    d = b.emit(tir.OpKind.CALL, result_ty=_i32(),
               attrs={"target": "helper"})
    e = b.emit(tir.OpKind.FFI_CALL, result_ty=_i32(),
               attrs={"target": "ext"})
    b.ret(b.add(d, e))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "declare i32 @ext()" in ll, ll
    assert "declare i32 @helper" not in ll, ll  # direct call: no declare
    assert f"%v{d.id} = call i32 @helper()" in ll, ll
    assert f"%v{e.id} = call i32 @ext()" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage205_ffi_call_emit_is_deterministic():
    """Two emits of a module with an FFI_CALL are byte-identical."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        r = b.emit(tir.OpKind.FFI_CALL, b.const_int(3),
                   result_ty=_i32(), attrs={"target": "ext"})
        b.ret(r)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage205_rejects_ffi_call_missing_target():
    """An FFI_CALL with no 'target' attr is rejected."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.FFI_CALL, result_ty=_i32(), attrs={})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="non-empty string 'target'"):
        llvm_ir.emit_module(mod)


def test_stage205_rejects_ffi_inconsistent_signature():
    """The same extern symbol called with two different signatures is
    rejected — an extern has exactly one signature."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.FFI_CALL, b.const_int(1), result_ty=_i32(),
           attrs={"target": "ext"})           # ext(i32)
    r = b.emit(tir.OpKind.FFI_CALL, result_ty=_i32(),
               attrs={"target": "ext"})       # ext() — differs
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="two different signatures"):
        llvm_ir.emit_module(mod)


def test_stage205_rejects_ffi_symbol_collides_with_defined_fn():
    """An FFI symbol that is also a defined function name is rejected
    — a `declare` cannot share a name with a `define`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("helper", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.FFI_CALL, result_ty=_i32(),
               attrs={"target": "helper"})    # collides with the define
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="also a defined function"):
        llvm_ir.emit_module(mod)
