"""Tests for helixc.backend.llvm_ir — v3.0 Phase D (Stages 200-206):
the additive LLVM IR emitter.

The emitter consumes the same host IR (`tir.Module`) that
`x86_64.py::compile_module_to_elf` consumes. Covered so far: the
scalar core (module header / target triple, `define`, integer
constants, add/sub/mul, `ret`); control flow (multi-block, `br`,
`phi`); the scalar op set (the six comparisons, `select`, `neg`,
div/mod, bitwise ops, unsigned integer dtypes); mutable local
variables and stack arrays (`alloca`/`load`/`store`/`getelementptr`);
direct + FFI function calls (`call`, with a module-scope `declare`
for FFI targets); and the Result<T,E> packed-tag intrinsics
(RESULT_PACK / RESULT_TAG / RESULT_PAYLOAD); panic (TRAP);
string-literal access (STR_PTR / STR_BYTE); and string output
(print_str PRINT). Everything else must be REJECTED loudly with
`LLVMEmitError` — a partial backend fails closed, it never emits
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


# ==========================================================================
# Stage 206 chunk A — Result<T,E> packed-tag intrinsics
# ==========================================================================
def test_stage206_emit_result_pack():
    """RESULT_PACK lowers to zext/shl + zext + or — a Result is one
    i64 with the tag in the high 32 bits and the payload in the low
    32."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.RESULT_PACK, b.const_int(0), b.const_int(42),
               result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id}.t0 = zext i32 0 to i64" in ll, ll
    assert f"%v{r.id}.t1 = shl i64 %v{r.id}.t0, 32" in ll, ll
    assert f"%v{r.id}.t2 = zext i32 42 to i64" in ll, ll
    assert f"%v{r.id} = or i64 %v{r.id}.t1, %v{r.id}.t2" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_emit_result_tag():
    """RESULT_TAG lowers to `lshr i64 ..., 32` then `trunc to i32` —
    the tag occupies the high 32 bits of the packed Result."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("p", tir.TIRScalar("i64"))],
                          tir.TIRScalar("i32"))
    r = b.emit(tir.OpKind.RESULT_TAG, fn.params[0],
               result_ty=tir.TIRScalar("i32"))
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id}.t0 = lshr i64 %v{fn.params[0].id}, 32" in ll, ll
    assert f"%v{r.id} = trunc i64 %v{r.id}.t0 to i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_emit_result_payload():
    """RESULT_PAYLOAD lowers to a single `trunc i64 ... to i32` — the
    payload is the low 32 bits of the packed Result."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("p", tir.TIRScalar("i64"))],
                          tir.TIRScalar("i32"))
    r = b.emit(tir.OpKind.RESULT_PAYLOAD, fn.params[0],
               result_ty=tir.TIRScalar("i32"))
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = trunc i64 %v{fn.params[0].id} to i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_result_pack_then_tag():
    """A realistic pattern — pack a Result, then read its tag back."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i32"))
    packed = b.emit(tir.OpKind.RESULT_PACK, b.const_int(1),
                    b.const_int(7), result_ty=tir.TIRScalar("i64"))
    tag = b.emit(tir.OpKind.RESULT_TAG, packed,
                 result_ty=tir.TIRScalar("i32"))
    b.ret(tag)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{packed.id} = or i64" in ll, ll
    assert f"%v{tag.id}.t0 = lshr i64 %v{packed.id}, 32" in ll, ll
    assert f"%v{tag.id} = trunc i64 %v{tag.id}.t0 to i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_result_pack_then_payload():
    """Pack a Result, then read its payload back."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i32"))
    packed = b.emit(tir.OpKind.RESULT_PACK, b.const_int(0),
                    b.const_int(99), result_ty=tir.TIRScalar("i64"))
    pay = b.emit(tir.OpKind.RESULT_PAYLOAD, packed,
                 result_ty=tir.TIRScalar("i32"))
    b.ret(pay)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{pay.id} = trunc i64 %v{packed.id} to i32" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_result_ops_emit_is_deterministic():
    """Two emits of a module with the Result intrinsics are
    byte-identical (the `%vN.tK` temp names derive from the result
    id)."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], tir.TIRScalar("i32"))
        packed = b.emit(tir.OpKind.RESULT_PACK, b.const_int(1),
                        b.const_int(5), result_ty=tir.TIRScalar("i64"))
        r = b.emit(tir.OpKind.RESULT_PAYLOAD, packed,
                   result_ty=tir.TIRScalar("i32"))
        b.ret(r)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206_rejects_result_pack_non_i64_result():
    """RESULT_PACK must produce an i64 — a packed Result is an i64."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.RESULT_PACK, b.const_int(0), b.const_int(1),
               result_ty=_i32())   # i32 result, must be i64
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="packed Result is an i64"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_result_pack_non_i32_operand():
    """RESULT_PACK's tag and payload must both be i32."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    bad = b.const_int(1, dtype="i64")   # payload i64, must be i32
    r = b.emit(tir.OpKind.RESULT_PACK, b.const_int(0), bad,
               result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="must be i32"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_result_tag_non_i64_operand():
    """RESULT_TAG's operand must be the i64 packed Result."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("p", _i32())], tir.TIRScalar("i32"))
    r = b.emit(tir.OpKind.RESULT_TAG, fn.params[0],
               result_ty=tir.TIRScalar("i32"))   # operand i32, must be i64
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="packed Result is an i64"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_result_payload_non_i32_result():
    """RESULT_PAYLOAD must produce an i32."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("p", tir.TIRScalar("i64"))],
                          tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.RESULT_PAYLOAD, fn.params[0],
               result_ty=tir.TIRScalar("i64"))   # i64 result, must be i32
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="tag/payload is an i32"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 206 chunk B — TRAP (panic)
# ==========================================================================
def test_stage206_emit_trap():
    """TRAP lowers to a `write(2, msg, len)` of the panic message, a
    `call exit`, and `unreachable` — plus a private string global for
    the message."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "oh no", "trap_id": 28501})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "call i64 @write(i32 2, ptr @.helix.str." in ll, ll
    assert "call void @exit(i32 " in ll, ll
    assert "unreachable" in ll, ll
    assert "private unnamed_addr constant" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_trap_message_format():
    """The panic message is `panic[<trap_id>]: <text>\\n`,
    byte-identical to x86_64.py — the parity gate compares stderr."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "boom", "trap_id": 12345})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # "panic[12345]: boom\n" — 19 bytes, the \n hex-escaped \0A.
    assert '[19 x i8] c"panic[12345]: boom\\0A"' in ll, ll


def test_stage206_trap_default_trap_id():
    """A TRAP with no 'trap_id' attr uses the default 28501; the exit
    status is its low byte (28501 & 0xFF == 85)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "x"})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "panic[28501]: x" in ll, ll
    assert "call void @exit(i32 85)" in ll, ll


def test_stage206_trap_exit_status_is_low_byte():
    """exit() receives the low 8 bits of the trap id — matching
    x86_64.py (the kernel truncates the exit status to a byte)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "x", "trap_id": 0x12AB})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"call void @exit(i32 {0x12AB & 0xFF})" in ll, ll  # 171


def test_stage206_trap_string_escaping():
    """A message byte that is not printable ASCII, or is `\"` / `\\`,
    is hex-escaped `\\XX` in the LLVM string constant."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": 'a"b\\c', "trap_id": 1})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "\\22" in ll and "\\5C" in ll and "\\0A" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_trap_string_deduped():
    """Two TRAPs with the same message share one string global."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("c", tir.TIRScalar("bool"))], _i32())
    t_blk = b.append_block()
    f_blk = b.append_block()
    b.emit(tir.OpKind.COND_BR, fn.params[0],
           attrs={"true_block": t_blk.id, "false_block": f_blk.id})
    b.switch_to(t_blk)
    b.emit(tir.OpKind.TRAP, attrs={"text": "same", "trap_id": 1})
    b.switch_to(f_blk)
    b.emit(tir.OpKind.TRAP, attrs={"text": "same", "trap_id": 1})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("private unnamed_addr constant") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_trap_declares_write_and_exit():
    """A module with a TRAP declares the externs `write` and `exit`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "x"})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "declare i64 @write(i32, ptr, i64)" in ll, ll
    assert "declare void @exit(i32)" in ll, ll


def test_stage206_trap_is_a_terminator():
    """TRAP terminates its block (it ends in `unreachable`) — a block
    consisting of just a TRAP is valid, with no separate RETURN."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "x"})
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "unreachable" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_trap_emit_is_deterministic():
    """Two emits of a module with a TRAP are byte-identical (the
    content-addressed string-global name is stable)."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        b.emit(tir.OpKind.TRAP, attrs={"text": "msg", "trap_id": 7})
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206_rejects_op_after_trap():
    """An op after a TRAP in the same block is rejected — TRAP is a
    terminator, so anything following it is unreachable."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "x"})
    b.ret(b.const_int(0))   # an op after the terminator
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="follows the block terminator"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_trap_with_operands():
    """TRAP takes no operands — the message is an attr, not an
    operand."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, b.const_int(0), attrs={"text": "x"})
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="TRAP takes no operands"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_trap_non_string_text():
    """TRAP's 'text' attr must be a string."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": 123})
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'text' attr"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_trap_non_int_trap_id():
    """TRAP's 'trap_id' attr must be an integer."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRAP, attrs={"text": "x", "trap_id": "oops"})
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="integer 'trap_id' attr"):
        llvm_ir.emit_module(mod)


def test_stage206_trap_result_not_referenceable():
    """TRAP defines no LLVM value — a result it carries (SSA
    bookkeeping only) is not registered, so a stray reference to it
    fails closed in `_ref` rather than emitting a dangling register."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.TRAP, result_ty=_i32(), attrs={"text": "x"})
    second = b.append_block()
    b.switch_to(second)
    b.ret(r)   # references the TRAP's (undefined) result
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="defined by no op"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 206 chunk C — string-literal access (STR_PTR / STR_BYTE)
# ==========================================================================
def test_stage206_emit_str_ptr():
    """STR_PTR lowers to `ptrtoint` of the literal's module-scope
    string constant — the literal's address as a u64."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("u64"))
    r = b.emit(tir.OpKind.STR_PTR, result_ty=tir.TIRScalar("u64"),
               attrs={"text": "hello"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = ptrtoint ptr @.helix.str." in ll, ll
    assert " to i64" in ll, ll
    assert "private unnamed_addr constant" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_emit_str_byte():
    """STR_BYTE lowers to a bounds-checked indexed byte load — an
    out-of-range index yields 0, with no out-of-bounds read (the
    clamp + the NUL-padded global)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], _i32())
    r = b.emit(tir.OpKind.STR_BYTE, fn.params[0], result_ty=_i32(),
               attrs={"text": "abc"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    p = fn.params[0].id
    assert f"%v{r.id}.t0 = icmp ult i32 %v{p}, 3" in ll, ll
    assert (f"%v{r.id}.t1 = select i1 %v{r.id}.t0, i32 %v{p}, "
            f"i32 0") in ll, ll
    assert (f"%v{r.id}.t2 = getelementptr [4 x i8], ptr @.helix.str."
            in ll), ll
    assert f"%v{r.id}.t3 = load i8, ptr %v{r.id}.t2" in ll, ll
    assert f"%v{r.id}.t4 = zext i8 %v{r.id}.t3 to i32" in ll, ll
    assert (f"%v{r.id} = select i1 %v{r.id}.t0, i32 %v{r.id}.t4, "
            f"i32 0") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_str_byte_empty_literal():
    """STR_BYTE on an empty string literal is valid — the NUL-padded
    global is `[1 x i8]`, so the bounds-clamped GEP never reads out of
    bounds; every index is out of range and yields 0."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], _i32())
    r = b.emit(tir.OpKind.STR_BYTE, fn.params[0], result_ty=_i32(),
               attrs={"text": ""})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"icmp ult i32 %v{fn.params[0].id}, 0" in ll, ll
    assert "[1 x i8]" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_str_ptr_and_byte_share_text():
    """STR_PTR and STR_BYTE on the same literal register two distinct
    globals — STR_PTR the exact bytes, STR_BYTE the NUL-padded form."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], tir.TIRScalar("u64"))
    b.emit(tir.OpKind.STR_BYTE, fn.params[0], result_ty=_i32(),
           attrs={"text": "hi"})
    p = b.emit(tir.OpKind.STR_PTR, result_ty=tir.TIRScalar("u64"),
               attrs={"text": "hi"})
    b.ret(p)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "[2 x i8]" in ll, ll   # STR_PTR — the exact bytes
    assert "[3 x i8]" in ll, ll   # STR_BYTE — + the NUL pad
    assert ll.count("private unnamed_addr constant") == 2, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_str_ptr_deduped():
    """Two STR_PTRs on the same literal share one string global."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("u64"))
    b.emit(tir.OpKind.STR_PTR, result_ty=tir.TIRScalar("u64"),
           attrs={"text": "dup"})
    p = b.emit(tir.OpKind.STR_PTR, result_ty=tir.TIRScalar("u64"),
               attrs={"text": "dup"})
    b.ret(p)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("private unnamed_addr constant") == 1, ll


def test_stage206_str_emit_is_deterministic():
    """Two emits of a module with STR_BYTE are byte-identical."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        fn = b.begin_function("f", [("i", _i32())], _i32())
        r = b.emit(tir.OpKind.STR_BYTE, fn.params[0], result_ty=_i32(),
                   attrs={"text": "xyz"})
        b.ret(r)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206_rejects_str_ptr_with_operands():
    """STR_PTR takes no operands — the literal is an attr."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("u64"))
    p = b.emit(tir.OpKind.STR_PTR, b.const_int(0),
               result_ty=tir.TIRScalar("u64"), attrs={"text": "x"})
    b.ret(p)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="STR_PTR takes no operands"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_str_ptr_non_i64_result():
    """STR_PTR must produce a u64 (i64) — it is a raw pointer."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    p = b.emit(tir.OpKind.STR_PTR, result_ty=_i32(),
               attrs={"text": "x"})   # i32 result, must be i64
    b.ret(p)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string pointer is a u64"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_str_byte_non_i32_result():
    """STR_BYTE must produce an i32."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.STR_BYTE, fn.params[0],
               result_ty=tir.TIRScalar("i64"),   # i64 result, must be i32
               attrs={"text": "x"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string byte is an i32"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_str_byte_non_string_text():
    """STR_BYTE's 'text' attr must be a string."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], _i32())
    r = b.emit(tir.OpKind.STR_BYTE, fn.params[0], result_ty=_i32(),
               attrs={"text": 99})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'text' attr"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 206 chunk D — string output (print_str PRINT)
# ==========================================================================
def test_stage206_emit_print_str():
    """A print_str PRINT lowers to `write(1, msg, len)` of a
    module-scope string constant, the i64 byte count truncated to the
    i32 result."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"text": "hi\n"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id}.t0 = call i64 @write(i32 1, ptr @.helix.str."
            in ll), ll
    assert f"%v{r.id} = trunc i64 %v{r.id}.t0 to i32" in ll, ll
    assert "private unnamed_addr constant" in ll, ll
    assert "declare i64 @write(i32, ptr, i64)" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_print_str_result_feeds_op():
    """A print_str PRINT's result (the i32 byte count) is a normal
    SSA value — it can feed a later op."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    n = b.emit(tir.OpKind.PRINT, result_ty=_i32(), attrs={"text": "x"})
    s = b.add(n, b.const_int(1))
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{s.id} = add i32 %v{n.id}, 1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_print_str_deduped():
    """Two print_str PRINTs of the same text share one string
    global."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(), attrs={"text": "msg"})
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"text": "msg"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("private unnamed_addr constant") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_print_emit_is_deterministic():
    """Two emits of a module with a print_str PRINT are
    byte-identical."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
                   attrs={"text": "out"})
        b.ret(r)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


# ==========================================================================
# Stage 206-R chunk — print_int PRINT lowers via the internal
# `@__helix_print_int(i32) -> i32` helper (digit-loop + write syscall).
# ==========================================================================
def test_stage206r_emit_print_int_call():
    """A print_int PRINT lowers to `call i32 @__helix_print_int(i32 %v)`
    — the heavy lifting (digit conversion + write syscall) lives in the
    helper, not at the call site."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @__helix_print_int(i32 %v"
            in ll), ll
    # The helper definition must be emitted exactly once.
    assert ll.count("define internal i32 @__helix_print_int(") == 1, ll
    # `write` declare comes through the helper's transitive FFI
    # registration (so a separate print_str in the same module
    # dedups to the same declare — see the test below).
    assert "declare i64 @write(i32, ptr, i64)" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_print_int_helper_emitted_once_per_module():
    """Two print_int PRINTs share ONE helper definition (the helper is
    deduplicated across the per-function helper sets in
    `emit_module`)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn1 = b.begin_function("a", [("n", _i32())], _i32())
    r1 = b.emit(tir.OpKind.PRINT, fn1.params[0], result_ty=_i32(),
                attrs={"_kind": "print_int"})
    b.ret(r1)
    b.end_function()
    fn2 = b.begin_function("c", [("m", _i32())], _i32())
    r2 = b.emit(tir.OpKind.PRINT, fn2.params[0], result_ty=_i32(),
                attrs={"_kind": "print_int"})
    b.ret(r2)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("define internal i32 @__helix_print_int(") == 1, ll
    # Two call sites, one declare.
    assert ll.count("call i32 @__helix_print_int(") == 2, ll
    assert ll.count("declare i64 @write(") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_print_int_and_print_str_dedup_write_declare():
    """A print_int (via the helper) and a print_str (which calls
    @write directly) share ONE `declare i64 @write(...)` — the helper's
    transitive FFI registration goes through the same
    `_register_ffi_declare` plumbing the direct call uses."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(), attrs={"text": "x"})
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("declare i64 @write(i32, ptr, i64)") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_print_int_result_feeds_op():
    """A print_int PRINT's result (the i32 byte count) is a normal SSA
    value — it can feed a later op."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    n = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    s = b.add(n, b.const_int(1))
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{s.id} = add i32 %v{n.id}, 1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_print_int_is_deterministic():
    """Two emits of the same print_int module are byte-identical
    (helper-set is iterated in sorted order, so the per-module helper
    block is stable)."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        fn = b.begin_function("f", [("n", _i32())], _i32())
        r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
                   attrs={"_kind": "print_int"})
        b.ret(r)
        b.end_function()
        return mod

    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_print_int_rejects_zero_operands():
    """A print_int PRINT requires its value operand — no operand is
    malformed and must fail closed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="print_int PRINT takes one operand"):
        llvm_ir.emit_module(mod)


def test_stage206r_print_int_rejects_two_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32()), ("b", _i32())], _i32())
    r = b.emit(tir.OpKind.PRINT, fn.params[0], fn.params[1],
               result_ty=_i32(), attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="print_int PRINT takes one operand"):
        llvm_ir.emit_module(mod)


def test_stage206r_print_int_rejects_non_i32_operand():
    """The helper signature is `(i32) -> i32` — a non-i32 operand
    cannot be forwarded directly and must fail closed (no silent
    truncation / extension)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", tir.TIRScalar("i64"))],
                          _i32())
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="print_int operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_print_int_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.PRINT, fn.params[0],
               result_ty=tir.TIRScalar("i64"),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="PRINT yields an i32"):
        llvm_ir.emit_module(mod)


def test_stage206r_helper_collision_with_user_function():
    """A user-defined function whose name collides with a reserved
    `__helix_` helper that the module actually uses must fail
    closed — silently shadowing the helper would emit a different
    body than the call expects."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    # A user function that incidentally chose the helper's name.
    b.begin_function("__helix_print_int", [("v", _i32())], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    # Another function actually calls a print_int PRINT, which
    # triggers the helper.
    fn = b.begin_function("f", [("n", _i32())], _i32())
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="collide with reserved"):
        llvm_ir.emit_module(mod)


def test_stage206r_helper_collision_only_when_used():
    """A user `__helix_print_int` function in a module that does NOT
    use the helper is left alone (the collision check is gated on
    actual helper use — no use, no collision)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("__helix_print_int", [("v", _i32())], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # The user's function is emitted as-is; no helper block was
    # added.
    assert "define i32 @__helix_print_int(" in ll, ll
    assert "define internal i32 @__helix_print_int(" not in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_print_int_helper_has_five_blocks():
    """Mock-shape check on the helper: its body should contain the
    five labelled basic blocks the digit-loop design needs (entry /
    loop / after_loop / prepend_sign / do_write). Caught a regression
    in chunk development where one block was missing terminator."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    for label in ("entry:", "loop:", "after_loop:",
                  "prepend_sign:", "do_write:"):
        assert f"\n{label}\n" in ll, (label, ll)


def test_stage206r_register_helper_function_rejects_unknown():
    """The `_register_helper_function` plumbing fails closed on an
    unknown helper name (defensive against future ops registering a
    typo or a name no longer in the registry)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    fn = next(iter(mod.functions.values()))
    emitter = llvm_ir._FnEmitter(fn)
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="internal helper 'no_such_helper' is "
                             "not registered"):
        emitter._register_helper_function("no_such_helper")


def test_stage206r_helper_function_table_drift_guard():
    """`_check_helper_function_table` rejects (1) a helper name not
    prefixed with `__helix_`, (2) a definition text whose declared
    function name does not match the registry key, (3) a definition
    text with no `ret` instruction, and (4) a definition text whose
    body does not call every callee its ffi_declares list claims."""
    original = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)

    def _restore() -> None:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original)

    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["bad_no_prefix"] = (
            llvm_ir._HelperFunctionSpec(
                definition=("define internal i32 "
                            "@bad_no_prefix() { ret i32 0 }"),
                ffi_declares=(),
            ))
        with pytest.raises(AssertionError, match="reserved `__helix_`"):
            llvm_ir._check_helper_function_table()
    finally:
        _restore()

    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_typo"] = (
            llvm_ir._HelperFunctionSpec(
                definition=("define internal i32 "
                            "@__helix_different() "
                            "{ ret i32 0 }"),
                ffi_declares=(),
            ))
        with pytest.raises(AssertionError,
                           match="does not match a `define internal"):
            llvm_ir._check_helper_function_table()
    finally:
        _restore()

    # No `ret` instruction in the helper body — caught by the drift
    # guard at module load (cheaper than waiting for mock_validate_ll
    # at first use).
    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_noret"] = (
            llvm_ir._HelperFunctionSpec(
                definition=("define internal void @__helix_noret() "
                            "{ unreachable }"),
                ffi_declares=(),
            ))
        with pytest.raises(AssertionError, match="no `ret` instruction"):
            llvm_ir._check_helper_function_table()
    finally:
        _restore()

    # ffi_declares mentions @write but the body never calls it.
    # Multi-line body so the `ret` line satisfies the line-scoped
    # ret-instruction guard (which fires first) — this test isolates
    # the registry-vs-body drift check.
    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_drift"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_drift() {\n"
                    "entry:\n"
                    "  ret i32 0\n"
                    "}"),
                ffi_declares=(
                    llvm_ir._FFIDeclareSpec(
                        target="write", callee="@write", ret_ty="i64",
                        arg_tys=("i32", "ptr", "i64")),
                ),
            ))
        with pytest.raises(AssertionError, match="registry and body"):
            llvm_ir._check_helper_function_table()
    finally:
        _restore()


def test_stage206r_helper_function_spec_is_frozen():
    """The `_HelperFunctionSpec` rejects attribute mutation — registry
    entries are module-scope constants. House style: frozen dataclass
    raises `FrozenInstanceError` (a subclass of `AttributeError`)."""
    spec = llvm_ir._HELPER_FUNCTIONS["__helix_print_int"]
    with pytest.raises(AttributeError):
        spec.definition = "tampered"  # type: ignore[misc]


def test_stage206r_helper_function_spec_is_final():
    """House style: cross-cutting backend result types are final
    (subclass-guarded) — see `Backend` and `ParityResult`."""
    with pytest.raises(TypeError, match="final"):
        class _Subclass(llvm_ir._HelperFunctionSpec):  # type: ignore[misc]
            pass


def test_stage206r_helper_function_spec_rejects_malformed_ffi():
    """The `_HelperFunctionSpec` rejects an ffi_declares entry that
    is not a `_FFIDeclareSpec` (a NamedTuple) — a bare 4-tuple
    rejected at construction so positional-order swaps cannot pass
    silently."""
    with pytest.raises(ValueError, match="must be a _FFIDeclareSpec"):
        llvm_ir._HelperFunctionSpec(
            definition=("define internal i32 @__helix_x() "
                        "{ ret i32 0 }"),
            ffi_declares=(("only", "four", "more", ("args",)),),  # type: ignore[arg-type]
        )


def test_stage206r_helper_public_registry_is_immutable():
    """The PUBLIC `_HELPER_FUNCTIONS` is a `MappingProxyType` view —
    callers cannot mutate it. Mutations must go through the private
    `_HELPER_FUNCTIONS_AUTHORITY` (this mirrors the MLIR backend's
    AUTHORITY pattern)."""
    with pytest.raises(TypeError):
        llvm_ir._HELPER_FUNCTIONS["__helix_tampered"] = (  # type: ignore[index]
            llvm_ir._HelperFunctionSpec(
                definition="define internal i32 @__helix_tampered() "
                           "{ ret i32 0 }",
                ffi_declares=(),
            ))


def test_stage206r_emit_module_rejects_helper_vs_ffi_collision():
    """Emit-time: a user `is_extern` declaration of `__helix_print_int`
    plus an FFI_CALL to it — combined with a module that triggers the
    helper — would otherwise emit BOTH a `declare` and a
    `define internal` for the same symbol (malformed LLVM IR that
    `mock_validate_ll` does not catch). The collision check in
    `emit_module` must fail closed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    # An `is_extern` function with the helper's name; an FFI_CALL
    # references it (so it lands in `declares`, not `defined`).
    b.begin_function("__helix_print_int", [("v", _i32())], _i32(),
                     attrs={"is_extern": True})
    b.end_function()
    fn = b.begin_function("caller", [("n", _i32())], _i32())
    b.emit(tir.OpKind.FFI_CALL, fn.params[0], result_ty=_i32(),
           attrs={"target": "__helix_print_int"})
    # Also emit a print_int PRINT in the SAME function — this is what
    # triggers the helper. (A clean module without print_int would
    # never trigger the helper, so there'd be no collision to detect.)
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="collide with reserved"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 206-R chunk — arena infrastructure + ARENA_PUSH op
# `@__helix_arena_base = internal global [CAP+1 x i32] zeroinitializer`
# (slot 0 = cursor, slots 1..CAP = user data; CAP = 2097152 matches
# x86_64.py). ARENA_PUSH lowers to `call i32 @__helix_arena_push(i32)`
# — a 4-block LLVM helper that bounds-checks the cursor, stores the
# value at the new slot, increments the cursor, returns the old cursor
# (or -1 on overflow). The arena global is emitted exactly once per
# module via the new `_MODULE_GLOBALS` registry.
# ==========================================================================
def test_stage206r_emit_arena_push_call():
    """ARENA_PUSH lowers to `call i32 @__helix_arena_push(i32 %v)`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @__helix_arena_push(i32 %v"
            in ll), ll
    # Helper defined exactly once.
    assert ll.count(
        "define internal i32 @__helix_arena_push(") == 1, ll
    # Arena global emitted exactly once.
    assert ll.count(
        "@__helix_arena_base = internal global") == 1, ll
    # The arena is sized to CAP + 1 slots (cursor + data).
    assert (f"[{llvm_ir._HELIX_ARENA_CAP + 1} x i32] "
            f"zeroinitializer") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_cap_matches_x86_backend():
    """The LLVM-side `_HELIX_ARENA_CAP` MUST match
    `x86_64.py::HELIX_ARENA_CAP` — Stage 207 parity gate would
    otherwise compare two backends with different overflow points
    (a silent divergence that the structural mock validator cannot
    detect)."""
    from helixc.backend import x86_64
    assert llvm_ir._HELIX_ARENA_CAP == x86_64.HELIX_ARENA_CAP


def test_stage206r_arena_push_helper_has_three_blocks():
    """The arena_push helper has three basic blocks (entry / in_bounds
    / exit) — the overflow case is folded into entry's direct branch
    to exit, so the phi at exit reads `[ -1, %entry ]`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    for label in ("entry:", "in_bounds:", "exit:"):
        assert f"\n{label}\n" in ll, (label, ll)
    # No `overflow:` block — the overflow path is entry's else branch.
    assert "\noverflow:\n" not in ll, ll


def test_stage206r_arena_push_helper_text_pinned():
    """The four parity-sensitive lines in the arena_push helper are
    pinned by exact substring: the overflow predicate, the +1
    arithmetic, the sign-extend width, and the GEP element type. A
    mutation of any one (e.g. `ugt` instead of `uge`, `zext` instead
    of `sext`, GEP on `i8`) would silently produce wrong IR that
    still passes the structural mock validator."""
    helper = llvm_ir._HELIX_ARENA_PUSH_HELPER
    assert (f"icmp uge i32 %cursor, {llvm_ir._HELIX_ARENA_CAP}"
            in helper), helper
    assert "%cursor_plus_one = add i32 %cursor, 1" in helper, helper
    assert ("%slot_idx_i64 = sext i32 %cursor_plus_one to i64"
            in helper), helper
    assert ("getelementptr inbounds i32, ptr @__helix_arena_base, "
            "i64 %slot_idx_i64") in helper, helper
    # The phi MUST return the OLD cursor on success, not the
    # incremented one — x86_64.py does `dec ecx` to recover the old
    # cursor for the same reason.
    assert ("phi i32 [ -1, %entry ], [ %cursor, %in_bounds ]"
            in helper), helper


def test_stage206r_arena_push_helper_emitted_once_per_module():
    """N ARENA_PUSH ops in M functions still emit ONE helper +
    ONE arena global."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn1 = b.begin_function("a", [("v", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH, fn1.params[0], result_ty=_i32())
    b.emit(tir.OpKind.ARENA_PUSH, fn1.params[0], result_ty=_i32())
    r1 = b.emit(tir.OpKind.ARENA_PUSH, fn1.params[0], result_ty=_i32())
    b.ret(r1)
    b.end_function()
    fn2 = b.begin_function("c", [("v", _i32())], _i32())
    r2 = b.emit(tir.OpKind.ARENA_PUSH, fn2.params[0], result_ty=_i32())
    b.ret(r2)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("define internal i32 @__helix_arena_push(") == 1, ll
    assert ll.count("@__helix_arena_base = internal global") == 1, ll
    assert ll.count("call i32 @__helix_arena_push(") == 4, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_push_module_global_not_emitted_when_unused():
    """A module with NO ARENA_PUSH ops must NOT emit the arena
    global (so a `print_int`-only program does not pay 8 MB of BSS
    for an unused arena)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("n", _i32())], _i32())
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "@__helix_arena_base" not in ll, ll
    # print_int's helper still emitted as before.
    assert "define internal i32 @__helix_print_int(" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_push_result_feeds_op():
    """ARENA_PUSH's result is a normal i32 SSA value."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", _i32())], _i32())
    idx = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    s = b.add(idx, b.const_int(1))
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{s.id} = add i32 %v{idx.id}, 1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_push_is_deterministic():
    """Two emits of the same ARENA_PUSH module are byte-identical
    (module-globals emitted in sorted-by-name order, like helpers)."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        fn = b.begin_function("f", [("v", _i32())], _i32())
        r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
        b.ret(r)
        b.end_function()
        return mod
    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_arena_push_rejects_zero_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH, result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH takes one operand"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_rejects_two_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32()), ("b", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH takes one operand"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_rejects_non_i32_operand():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", tir.TIRScalar("i64"))], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", _i32())], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0],
               result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_module_global_spec_is_frozen():
    spec = llvm_ir._MODULE_GLOBALS["__helix_arena_base"]
    with pytest.raises(AttributeError):
        spec.name = "tampered"  # type: ignore[misc]


def test_stage206r_module_global_spec_is_final():
    with pytest.raises(TypeError, match="final"):
        class _Subclass(llvm_ir._ModuleGlobalSpec):  # type: ignore[misc]
            pass


def test_stage206r_module_global_spec_rejects_non_helix_prefix():
    with pytest.raises(ValueError, match="reserved `__helix_` prefix"):
        llvm_ir._ModuleGlobalSpec(
            name="user_global",
            definition="@user_global = internal global i32 0",
        )


def test_stage206r_module_global_spec_rejects_name_vs_def_mismatch():
    with pytest.raises(ValueError,
                       match="registry name and global declaration"):
        llvm_ir._ModuleGlobalSpec(
            name="__helix_typo",
            definition="@__helix_different = internal global i32 0",
        )


def test_stage206r_module_global_public_registry_is_immutable():
    """`_MODULE_GLOBALS` is `MappingProxyType` — same pattern as
    `_HELPER_FUNCTIONS`."""
    with pytest.raises(TypeError):
        llvm_ir._MODULE_GLOBALS["__helix_tamper"] = (  # type: ignore[index]
            llvm_ir._ModuleGlobalSpec(
                name="__helix_tamper",
                definition="@__helix_tamper = internal global i32 0",
            ))


def test_stage206r_module_global_drift_guard_rejects_unknown_dep():
    """A helper that declares `module_globals=("__helix_unknown",)`
    must be rejected at module load by
    `_check_module_global_table` (cheaper than waiting for first
    use to fail in `_register_helper_function`)."""
    original_helpers = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_bad"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_bad() {\n"
                    "entry:\n"
                    "  ret i32 0\n"
                    "}"),
                ffi_declares=(),
                module_globals=("__helix_unknown",),
            ))
        with pytest.raises(AssertionError,
                           match="declares module-global dependency"):
            llvm_ir._check_module_global_table()
    finally:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original_helpers)


def test_stage206r_helper_spec_rejects_malformed_module_globals():
    with pytest.raises(ValueError, match="module_globals must be"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ffi_declares=(),
            module_globals="not_a_tuple",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError,
                       match="module_globals entry"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ffi_declares=(),
            module_globals=("user_global",),
        )


def test_stage206r_arena_push_helper_does_not_collide_with_print_int():
    """A module that uses BOTH ARENA_PUSH and a print_int PRINT emits
    BOTH helpers + the arena global + the @write FFI declare, with
    no cross-contamination."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    r = b.emit(tir.OpKind.PRINT, fn.params[0], result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "define internal i32 @__helix_arena_push(" in ll, ll
    assert "define internal i32 @__helix_print_int(" in ll, ll
    assert "@__helix_arena_base = internal global" in ll, ll
    assert "declare i64 @write(" in ll, ll
    # Sorted-by-name emission order: arena_push before print_int.
    arena_pos = ll.index("define internal i32 @__helix_arena_push(")
    print_pos = ll.index("define internal i32 @__helix_print_int(")
    assert arena_pos < print_pos, (arena_pos, print_pos)
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_global_rejects_user_function_collision():
    """A user-defined function named `__helix_arena_base` would
    shadow the arena global. The collision gate added in this chunk
    must fail closed when the global is actually used."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("__helix_arena_base", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    fn = b.begin_function("f", [("v", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="module-global name"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_global_rejects_user_ffi_collision():
    """Sibling of the helper-vs-FFI-declare collision test: a user
    `is_extern` declaration of `__helix_arena_base` + an FFI_CALL
    targeting it (so it lands in `declares`, not `defined`) +
    ARENA_PUSH (so the arena global is actually pulled in) must
    fail closed. Without this gate, `emit_module` would emit BOTH
    the user's `declare ... @__helix_arena_base(...)` AND the
    `@__helix_arena_base = internal global ...` line — malformed
    LLVM IR that `mock_validate_ll` does not detect."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    # `is_extern` user declaration of the arena symbol.
    b.begin_function("__helix_arena_base", [("v", _i32())], _i32(),
                     attrs={"is_extern": True})
    b.end_function()
    fn = b.begin_function("caller", [("v", _i32())], _i32())
    # FFI_CALL puts the extern in `declares`, not `defined`.
    b.emit(tir.OpKind.FFI_CALL, fn.params[0], result_ty=_i32(),
           attrs={"target": "__helix_arena_base"})
    # ARENA_PUSH pulls the arena global in (otherwise no collision
    # to detect).
    r = b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="module-global name"):
        llvm_ir.emit_module(mod)


def test_stage206r_module_global_drift_guard_rejects_helper_vs_global_name_collision():
    """Module-load drift guard: a `__helix_*` name registered in
    BOTH `_HELPER_FUNCTIONS_AUTHORITY` and `_MODULE_GLOBALS_AUTHORITY`
    would emit BOTH a `define internal i32 @X(...)` AND a
    `@X = internal global ...` for the same `@X` — malformed LLVM
    IR that `llvm-as` rejects with "redefinition of @X" and that
    `mock_validate_ll` does NOT detect."""
    original_helpers = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    original_globals = dict(llvm_ir._MODULE_GLOBALS_AUTHORITY)

    def _restore() -> None:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original_helpers)
        llvm_ir._MODULE_GLOBALS_AUTHORITY.clear()
        llvm_ir._MODULE_GLOBALS_AUTHORITY.update(original_globals)

    try:
        # Register the SAME `__helix_*` name as a helper AND a global.
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_collide"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_collide() {\n"
                    "entry:\n"
                    "  ret i32 0\n"
                    "}"),
                ffi_declares=(),
                module_globals=(),
            ))
        llvm_ir._MODULE_GLOBALS_AUTHORITY["__helix_collide"] = (
            llvm_ir._ModuleGlobalSpec(
                name="__helix_collide",
                definition="@__helix_collide = internal global i32 0",
            ))
        with pytest.raises(AssertionError,
                           match="appear in BOTH"):
            llvm_ir._check_module_global_table()
    finally:
        _restore()


def test_stage206r_module_global_spec_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        llvm_ir._ModuleGlobalSpec(
            name="",
            definition="@__helix_x = internal global i32 0",
        )


def test_stage206r_module_global_spec_rejects_empty_definition():
    with pytest.raises(ValueError, match="non-empty string"):
        llvm_ir._ModuleGlobalSpec(
            name="__helix_x",
            definition="",
        )


def test_stage206r_helper_spec_rejects_module_globals_duplicates():
    """`module_globals` is consumed set-like at emit time, but a
    duplicate at the spec level is a typo / drift signal — reject
    at construction (mirrors `Backend.required_dialects`)."""
    with pytest.raises(ValueError, match="module_globals has duplicates"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ffi_declares=(),
            module_globals=("__helix_arena_base", "__helix_arena_base"),
        )


# ==========================================================================
# Stage 206-R chunk — ARENA_GET / ARENA_SET / ARENA_LEN
# Three more arena ops, each its own internal helper (GET / SET both
# 3-block bounds-checked; LEN single-load). All four arena helpers
# (push / get / set / len) share the `@__helix_arena_base` global via
# the `_MODULE_GLOBALS` registry.
# ==========================================================================
def test_stage206r_emit_arena_get_call():
    """ARENA_GET lowers to `call i32 @__helix_arena_get(i32 %idx)`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_GET, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @__helix_arena_get(i32 %v"
            in ll), ll
    assert ll.count(
        "define internal i32 @__helix_arena_get(") == 1, ll
    assert ll.count(
        "@__helix_arena_base = internal global") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_emit_arena_set_call_with_result():
    """ARENA_SET with a result slot lowers to a value-returning call
    (helper always returns i32 0). Two i32 operands."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32()), ("v", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @__helix_arena_set(i32 %v0, "
            f"i32 %v1)" in ll), ll
    assert ll.count(
        "define internal i32 @__helix_arena_set(") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_emit_arena_set_call_without_result():
    """ARENA_SET with no result lowers to a discard call (helper's
    return value drops on the floor — TIR-canonical shape)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32()), ("v", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1])
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # No `%vN =` prefix — call's return is discarded.
    assert ("call i32 @__helix_arena_set(i32 %v0, i32 %v1)"
            in ll), ll
    # Sanity: ensure we did NOT accidentally bind a result.
    assert "= call i32 @__helix_arena_set" not in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_emit_arena_len_call():
    """ARENA_LEN lowers to `call i32 @__helix_arena_len()` (no
    operands)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.ARENA_LEN, result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = call i32 @__helix_arena_len()" in ll, ll
    assert ll.count("define internal i32 @__helix_arena_len(") == 1, ll
    assert "load i32, ptr @__helix_arena_base, align 4" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_four_arena_ops_share_one_global():
    """A module that touches all four arena ops emits exactly ONE
    arena global + the four helpers + four call sites."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32()), ("v", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH, fn.params[1], result_ty=_i32())
    b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1])
    g = b.emit(tir.OpKind.ARENA_GET, fn.params[0], result_ty=_i32())
    length = b.emit(tir.OpKind.ARENA_LEN, result_ty=_i32())
    s = b.add(g, length)
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("@__helix_arena_base = internal global") == 1, ll
    for name in ("__helix_arena_push", "__helix_arena_get",
                 "__helix_arena_set", "__helix_arena_len"):
        assert ll.count(f"define internal i32 @{name}(") == 1, (name, ll)
        assert ll.count(f"call i32 @{name}(") == 1, (name, ll)
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_get_helper_returns_zero_on_overflow():
    """The arena_get helper's phi reads `[ 0, %entry ]` so an
    out-of-bounds index returns 0 (matches x86_64.py)."""
    helper = llvm_ir._HELIX_ARENA_GET_HELPER
    assert (f"icmp uge i32 %idx, {llvm_ir._HELIX_ARENA_CAP}"
            in helper), helper
    assert ("phi i32 [ 0, %entry ], [ %loaded, %in_bounds ]"
            in helper), helper
    # Idx+1 + sext to i64 + GEP i32 — must mirror push's arithmetic
    # so a slot stored by push is readable by get at the same index.
    assert "%idx_plus_one = add i32 %idx, 1" in helper, helper
    assert "%idx_i64 = sext i32 %idx_plus_one to i64" in helper, helper
    assert ("getelementptr inbounds i32, ptr "
            "@__helix_arena_base, i64 %idx_i64") in helper, helper


def test_stage206r_arena_set_helper_silently_skips_overflow():
    """The arena_set helper branches to exit on overflow WITHOUT
    storing — matches x86_64.py's "out-of-bounds set silently
    no-ops" comment (line 3285). Always returns i32 0 so the op
    handler can bind the result uniformly."""
    helper = llvm_ir._HELIX_ARENA_SET_HELPER
    assert (f"icmp uge i32 %idx, {llvm_ir._HELIX_ARENA_CAP}"
            in helper), helper
    # `br i1 %ovfl, label %exit, label %in_bounds` — overflow goes
    # directly to exit (no store). The store lives only in
    # in_bounds.
    assert "br i1 %ovfl, label %exit, label %in_bounds" in helper, helper
    assert "store i32 %value, ptr %slot_ptr, align 4" in helper, helper
    # The exit block is the only `ret`.
    assert helper.count("ret i32") == 1, helper
    assert "ret i32 0" in helper, helper


def test_stage206r_arena_len_helper_is_single_load():
    """arena_len is the smallest possible helper — entry block with
    one load + ret."""
    helper = llvm_ir._HELIX_ARENA_LEN_HELPER
    assert "load i32, ptr @__helix_arena_base, align 4" in helper, helper
    assert helper.count("\nentry:") == 1, helper
    # No additional labelled blocks (no `\n<word>:` for non-entry).
    block_lines = [
        ln for ln in helper.splitlines()
        if ln.endswith(":") and not ln.lstrip().startswith(";")
    ]
    assert block_lines == ["entry:"], block_lines


def test_stage206r_arena_get_rejects_zero_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.ARENA_GET, result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_GET takes one operand"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_get_rejects_non_i32_operand():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", tir.TIRScalar("i64"))], _i32())
    r = b.emit(tir.OpKind.ARENA_GET, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_GET operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_get_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.ARENA_GET, fn.params[0],
               result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_GET result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_set_rejects_one_operand():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("i", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_SET, fn.params[0])
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_SET takes two operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_set_rejects_three_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32()), ("c", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1], fn.params[2])
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_SET takes two operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_set_rejects_non_i32_index():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("i", tir.TIRScalar("i64")), ("v", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1])
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_SET index has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_set_rejects_non_i32_value():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("i", _i32()), ("v", tir.TIRScalar("i64"))], _i32())
    b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1])
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_SET value has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_len_rejects_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("x", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_LEN, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_LEN takes no operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_len_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.ARENA_LEN, result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_LEN result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_all_arena_ops_are_deterministic():
    """A module using all four arena ops emits byte-identically
    twice (the sorted-by-name helper emission keeps the helper
    block order stable: get / len / push / set)."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        fn = b.begin_function("f", [("i", _i32()), ("v", _i32())], _i32())
        b.emit(tir.OpKind.ARENA_PUSH, fn.params[1], result_ty=_i32())
        b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1])
        g = b.emit(tir.OpKind.ARENA_GET, fn.params[0], result_ty=_i32())
        length = b.emit(tir.OpKind.ARENA_LEN, result_ty=_i32())
        s = b.add(g, length)
        b.ret(s)
        b.end_function()
        return mod
    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_arena_set_rejects_non_i32_result():
    """When ARENA_SET carries an optional result, it must be i32 —
    the helper always returns i32 0. Symmetry with the GET/LEN
    result-type rejection paths."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("i", _i32()), ("v", _i32())], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1],
               result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_SET result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_set_rejects_two_results():
    """ARENA_SET tolerates 0 or 1 result; 2+ is malformed. Pin the
    `expects zero or one results` diagnostic so a future refactor
    that loosens this guard fails the test."""
    # The TIR builder normally enforces single-result; construct the
    # op directly to test the LLVM-side guard.
    op = tir.Op(
        kind=tir.OpKind.ARENA_SET,
        operands=[
            tir.Value(id=0, ty=_i32()),
            tir.Value(id=1, ty=_i32()),
        ],
        results=[
            tir.Value(id=10, ty=_i32()),
            tir.Value(id=11, ty=_i32()),  # the extra one — malformed
        ],
    )
    fn = tir.FnIR(
        name="f",
        params=[
            tir.Value(id=0, ty=_i32()),
            tir.Value(id=1, ty=_i32()),
        ],
        return_ty=_i32(),
        blocks=[tir.Block(id=0, params=[], ops=[op])],
    )
    emitter = llvm_ir._FnEmitter(fn)
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="expects zero or one results"):
        emitter._emit_op(op)


# ==========================================================================
# Stage 206-R chunk — write_file PRINT lowers inline to the libc
# sequence open(path, O_WRONLY|O_CREAT|O_TRUNC, 0644) -> write(fd,
# content, len) -> close(fd); result is `nwritten < 0 ? nwritten : 0`
# (negative on failure, 0 on success — matches x86_64.py).
# ==========================================================================
def test_stage206r_emit_write_file():
    """A write_file PRINT lowers to the open/write/close libc
    sequence, with the path and content stored as module-scope
    string constants (path NUL-terminated, content raw)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/x",
                      "content": "data"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # The path constant is NUL-terminated; content is raw.
    assert "[7 x i8] c\"/tmp/x\\00\"" in ll, ll
    assert "[4 x i8] c\"data\"" in ll, ll
    # The three libc declares.
    assert "declare i32 @open(ptr, i32, i32)" in ll, ll
    assert "declare i64 @write(i32, ptr, i64)" in ll, ll
    assert "declare i32 @close(i32)" in ll, ll
    # The six-instruction call site. Constants: 577 =
    # O_WRONLY|O_CREAT|O_TRUNC, 420 = 0o644.
    assert f"%v{r.id}.fd = call i32 @open(ptr @.helix.str." in ll, ll
    assert "i32 577, i32 420)" in ll, ll
    assert (f"%v{r.id}.nwritten = call i64 @write(i32 %v{r.id}.fd, "
            f"ptr @.helix.str." in ll), ll
    assert "i64 4)" in ll, ll  # content length
    assert f"%v{r.id}.close = call i32 @close(i32 %v{r.id}.fd)" in ll, ll
    assert (f"%v{r.id}.nw32 = trunc i64 %v{r.id}.nwritten to i32"
            in ll), ll
    assert (f"%v{r.id}.is_neg = icmp slt i32 %v{r.id}.nw32, 0"
            in ll), ll
    assert (f"%v{r.id} = select i1 %v{r.id}.is_neg, i32 "
            f"%v{r.id}.nw32, i32 0") in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_write_file_dedups_path_and_content():
    """Two write_file PRINTs with the same path AND same content
    share ONE path global and ONE content global (string constants
    are content-addressed via SHA-256)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(),
           attrs={"_kind": "write_file", "path": "/tmp/a",
                  "content": "x"})
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/a",
                      "content": "x"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # path "/tmp/a\\0" + content "x" = 2 string constants, exactly.
    assert ll.count("private unnamed_addr constant") == 2, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_write_file_distinct_globals_for_different_strings():
    """Two write_file PRINTs with DIFFERENT path or content each get
    their own globals (no false-dedup; the NUL terminator on the
    path is what distinguishes a 'path' from a 'content' that happens
    to spell the same text)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(),
           attrs={"_kind": "write_file", "path": "/tmp/a",
                  "content": "one"})
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/b",
                      "content": "two"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # 4 distinct string constants (2 paths + 2 contents).
    assert ll.count("private unnamed_addr constant") == 4, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_write_file_dedups_libc_declares():
    """Two write_file PRINTs in one module emit ONE declare per libc
    symbol (open / write / close), regardless of how many call sites
    reference them."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(),
           attrs={"_kind": "write_file", "path": "/tmp/a",
                  "content": "x"})
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/b",
                      "content": "y"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("declare i32 @open(") == 1, ll
    assert ll.count("declare i64 @write(") == 1, ll
    assert ll.count("declare i32 @close(") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_write_file_write_declare_dedups_with_print_str():
    """A write_file (which calls @write) and a print_str (which also
    calls @write) share ONE `declare i64 @write(...)` — both go
    through `_register_ffi_declare` with identical signatures."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(), attrs={"text": "x"})
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/a",
                      "content": "data"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("declare i64 @write(i32, ptr, i64)") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_write_file_result_feeds_op():
    """A write_file PRINT's result is a normal i32 SSA value — can
    feed a later op."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    n = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/x",
                      "content": "y"})
    s = b.add(n, b.const_int(1))
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{s.id} = add i32 %v{n.id}, 1" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_write_file_is_deterministic():
    """Two emits of the same write_file module are byte-identical."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("f", [], _i32())
        r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
                   attrs={"_kind": "write_file", "path": "/tmp/x",
                          "content": "data"})
        b.ret(r)
        b.end_function()
        return mod
    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_write_file_rejects_operands():
    """A write_file PRINT takes no operands — fail closed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, b.const_int(0), result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/x",
                      "content": "y"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="write_file PRINT takes no operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_write_file_rejects_missing_path_attr():
    """The `path` attr is required — a missing one fails closed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "content": "y"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'path' attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_write_file_rejects_missing_content_attr():
    """The `content` attr is required — a missing one fails closed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/x"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'content' attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_write_file_rejects_non_string_path():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": 42,
                      "content": "y"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'path' attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_write_file_rejects_non_string_content():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/x",
                      "content": b"bytes"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'content' attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_write_file_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.PRINT, result_ty=tir.TIRScalar("i64"),
               attrs={"_kind": "write_file", "path": "/tmp/x",
                      "content": "y"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="PRINT yields an i32"):
        llvm_ir.emit_module(mod)


def test_stage206r_write_file_empty_content():
    """A write_file with empty content still works: the content
    global is a zero-length array, len arg to write is 0."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/x",
                      "content": ""})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # length 0 in the write call.
    assert "i64 0)" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_write_file_rejects_embedded_nul_in_path():
    """An embedded NUL in the `path` attr would silently truncate the
    filesystem target — `open(2)` reads the path as a C-string and
    stops at the first NUL. Fail closed (audit HIGH-1)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file",
                      "path": "/tmp/a\x00/etc/passwd",
                      "content": "x"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="embedded NUL"):
        llvm_ir.emit_module(mod)


def test_stage206r_write_file_content_with_embedded_nul_preserved():
    """A `content` attr with embedded NUL bytes is preserved verbatim
    — `write(fd, ptr, len)` takes a length, not a NUL-terminated
    string, so binary data containing NULs is written intact.
    Locks the contract documented in the branch comment (audit LOW-3)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "write_file", "path": "/tmp/x",
                      "content": "a\x00b"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # 3 content bytes including the NUL — escaped as `\\00` in LLVM's
    # cstring form.
    assert "[3 x i8] c\"a\\00b\"" in ll, ll
    # The write call passes i64 3 (the full content length).
    assert f"%v{r.id}.nwritten = call i64 @write(" in ll, ll
    assert "i64 3)" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206_rejects_print_str_with_operands():
    """A print_str PRINT takes no operands — the text is an attr."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, b.const_int(0), result_ty=_i32(),
               attrs={"text": "x"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="print_str PRINT takes no operands"):
        llvm_ir.emit_module(mod)


def test_stage206_rejects_print_non_i32_result():
    """A print_str PRINT must yield an i32 byte count."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.PRINT, result_ty=tir.TIRScalar("i64"),
               attrs={"text": "x"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="PRINT yields an i32"):
        llvm_ir.emit_module(mod)


# ==========================================================================
# Stage 208 5-clean-gate fix — emit_module's function filter
# (is_extern declarations skipped; @kernel functions rejected loudly)
# ==========================================================================
def test_emit_module_skips_is_extern_function():
    """An `is_extern` ("extern C") function is a body-less declaration
    — `emit_module` emits NO `define` for it (mirroring x86_64.py),
    rather than handing the empty FnIR to `_FnEmitter` and raising a
    misleading "block has no terminator"."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("ext_decl", [("c", _i32())], _i32(),
                     attrs={"is_extern": True})
    b.end_function()
    b.begin_function("main", [], _i32())
    b.ret(b.const_int(42))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert llvm_ir.mock_validate_ll(ll) == [], ll
    assert "define i32 @main()" in ll
    # The uncalled extern gets neither a define nor a declare.
    assert "@ext_decl" not in ll, ll


def test_emit_module_rejects_kernel_function():
    """A `@kernel` (GPU) function is outside the LLVM host CPU
    backend's scope — `emit_module` rejects it with a loud, clear
    `LLVMEmitError` naming the kernel cause."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("gpu_k", [], _i32(), attrs={"kernel": True})
    b.end_function()
    b.begin_function("main", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError, match="@kernel"):
        llvm_ir.emit_module(mod)


def test_emit_module_extern_ffi_call_no_define_clash():
    """An extern-C symbol that is BOTH an `is_extern` FnIR in
    `module.functions` AND the target of an FFI_CALL must emit cleanly:
    the extern gets a `declare` (from the FFI_CALL), not a `define`,
    and is NOT mistaken for a defined function colliding with that
    declare."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("puts", [("s", _i32())], _i32(),
                     attrs={"is_extern": True})
    b.end_function()
    b.begin_function("main", [], _i32())
    r = b.emit(tir.OpKind.FFI_CALL, b.const_int(0), result_ty=_i32(),
               attrs={"target": "puts"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert llvm_ir.mock_validate_ll(ll) == [], ll
    lines = [ln.strip() for ln in ll.splitlines()]
    assert any(ln.startswith("declare") and "@puts" in ln
               for ln in lines), ll
    assert not any(ln.startswith("define") and "@puts" in ln
                   for ln in lines), ll
    assert "define i32 @main()" in ll
