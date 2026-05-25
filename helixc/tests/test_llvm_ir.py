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
                ret_ty="i32",
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
                ret_ty="i32",
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
                ret_ty="void",
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
                ret_ty="i32",
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
            ret_ty="i32",
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
                ret_ty="i32",
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
    detect).

    v3.1 step 6a note: both sides now source ARENA_CAP from
    `helixc.backend._shared_constants` so the equality is
    tautological — until the day someone introduces a private
    re-definition. `test_shared_constants.py` is the primary
    drift guard via source-grep; this remains as belt-and-braces."""
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
                ret_ty="i32",
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
            ret_ty="i32",
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
            ret_ty="i32",
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
                ret_ty="i32",
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
            ret_ty="i32",
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
# Stage 206-R chunk — ARENA_PUSH_PAIR / ARENA_PUSH_TRIPLE
# Atomic multi-slot pushes. PAIR writes 2 i32s at cursor+1/cursor+2
# with bounds threshold CAP-1; TRIPLE writes 3 i32s at
# cursor+1/cursor+2/cursor+3 with threshold CAP-2. Atomic-or-none:
# on overflow, neither/none of the writes happen AND the cursor does
# NOT advance. Returns the old cursor (= slot index of left) or -1
# on overflow.
# ==========================================================================
def test_stage206r_emit_arena_push_pair_call():
    """ARENA_PUSH_PAIR lowers to `call i32 @__helix_arena_push_pair(
    i32 %left, i32 %right)`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32()), ("b", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @__helix_arena_push_pair("
            f"i32 %v0, i32 %v1)" in ll), ll
    assert ll.count(
        "define internal i32 @__helix_arena_push_pair(") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_emit_arena_push_triple_call():
    """ARENA_PUSH_TRIPLE lowers to `call i32
    @__helix_arena_push_triple(i32 %left, i32 %middle, i32 %right)`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @__helix_arena_push_triple("
            f"i32 %v0, i32 %v1, i32 %v2)" in ll), ll
    assert ll.count(
        "define internal i32 @__helix_arena_push_triple(") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_push_pair_helper_bounds_threshold():
    """PAIR overflow threshold is `cursor >= CAP - 1` (so cursor+1
    and cursor+2 both fit). Matches x86_64.py::ARENA_PUSH_PAIR line
    3194-3195."""
    helper = llvm_ir._HELIX_ARENA_PUSH_PAIR_HELPER
    assert (f"icmp uge i32 %cursor, {llvm_ir._HELIX_ARENA_CAP - 1}"
            in helper), helper
    # cursor advances by 2 on success (atomic; mirrors x86 line 3206).
    assert ("store i32 %cursor_plus_two, ptr @__helix_arena_base"
            in helper), helper
    # Returns OLD cursor (= slot index of left) on success.
    assert ("phi i32 [ -1, %entry ], [ %cursor, %in_bounds ]"
            in helper), helper


def test_stage206r_arena_push_triple_helper_bounds_threshold():
    """TRIPLE overflow threshold is `cursor >= CAP - 2`. Matches
    x86_64.py::ARENA_PUSH_TRIPLE line 3239-3240."""
    helper = llvm_ir._HELIX_ARENA_PUSH_TRIPLE_HELPER
    assert (f"icmp uge i32 %cursor, {llvm_ir._HELIX_ARENA_CAP - 2}"
            in helper), helper
    # cursor advances by 3 on success.
    assert ("store i32 %cursor_plus_three, ptr @__helix_arena_base"
            in helper), helper
    # Three stores in in_bounds — atomic.
    assert helper.count("store i32 %left,") == 1, helper
    assert helper.count("store i32 %middle,") == 1, helper
    assert helper.count("store i32 %right,") == 1, helper
    assert ("phi i32 [ -1, %entry ], [ %cursor, %in_bounds ]"
            in helper), helper


def test_stage206r_arena_pair_triple_atomic_on_overflow():
    """Atomic-or-none: on overflow the helper branches DIRECTLY from
    entry to exit, skipping the entire in_bounds block (no partial
    writes, no cursor advance). Pin "no store outside in_bounds"
    against BOTH entry and exit so a future refactor cannot regress
    the contract by sinking a conditional cursor write into exit."""
    for helper in (llvm_ir._HELIX_ARENA_PUSH_PAIR_HELPER,
                   llvm_ir._HELIX_ARENA_PUSH_TRIPLE_HELPER):
        assert ("br i1 %ovfl, label %exit, label %in_bounds"
                in helper), helper
        # cursor write lives only in in_bounds, never in entry/exit
        # — pin this so a future refactor that "optimizes" by
        # storing the cursor up-front (and only conditionally
        # advancing) cannot regress the atomic-or-none contract.
        entry_block = helper.split("in_bounds:")[0]
        assert ("store i32" not in entry_block), entry_block
        exit_block = helper.split("exit:")[1]
        assert ("store i32" not in exit_block), exit_block


def test_stage206r_arena_push_pair_rejects_one_operand():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_PAIR takes two operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_pair_rejects_three_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_PAIR,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_PAIR takes two operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_pair_rejects_non_i32_left():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", tir.TIRScalar("i64")), ("b", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_PAIR left operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_pair_rejects_non_i32_right():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", tir.TIRScalar("i64"))], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_PAIR right operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_pair_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32())], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
               result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_PAIR result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_triple_rejects_two_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32()), ("b", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_TRIPLE takes three operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_triple_rejects_four_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32()),
              ("c", _i32()), ("d", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2], fn.params[3],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_TRIPLE takes three operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_triple_rejects_non_i32_left():
    """TRIPLE's per-operand check iterates `(left, middle, right)`;
    the LEFT case is the first iteration. Pin it explicitly so a
    future refactor that short-circuits the loop (e.g. starts at
    `middle`) cannot silently regress the left-position guard.
    Audit-fix: type-design MUST-FIX MEDIUM on this chunk."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", tir.TIRScalar("i64")), ("b", _i32()), ("c", _i32())],
        _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_TRIPLE left operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_triple_rejects_non_i32_middle():
    """TRIPLE checks ALL THREE operands individually; the middle one
    is the easiest to forget if the handler loops sloppily."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", tir.TIRScalar("i64")), ("c", _i32())],
        _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_TRIPLE middle operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_triple_rejects_non_i32_right():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32()), ("c", tir.TIRScalar("i64"))],
        _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_TRIPLE right operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_arena_push_triple_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32()), ("c", _i32())],
        tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=tir.TIRScalar("i64"))
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="ARENA_PUSH_TRIPLE result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_all_six_arena_ops_share_one_global():
    """All six arena ops (PUSH / GET / SET / LEN / PUSH_PAIR /
    PUSH_TRIPLE) sharing one global + the same overflow contract."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("i", _i32()), ("v", _i32()), ("w", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH, fn.params[1], result_ty=_i32())
    b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[1], fn.params[2],
           result_ty=_i32())
    b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
           fn.params[1], fn.params[2], fn.params[0], result_ty=_i32())
    b.emit(tir.OpKind.ARENA_SET, fn.params[0], fn.params[1])
    g = b.emit(tir.OpKind.ARENA_GET, fn.params[0], result_ty=_i32())
    length = b.emit(tir.OpKind.ARENA_LEN, result_ty=_i32())
    s = b.add(g, length)
    b.ret(s)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("@__helix_arena_base = internal global") == 1, ll
    for name in ("__helix_arena_push", "__helix_arena_push_pair",
                 "__helix_arena_push_triple", "__helix_arena_get",
                 "__helix_arena_set", "__helix_arena_len"):
        assert ll.count(f"define internal i32 @{name}(") == 1, (name, ll)
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_pair_triple_are_deterministic():
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        fn = b.begin_function(
            "f", [("a", _i32()), ("b", _i32()), ("c", _i32())], _i32())
        b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
               result_ty=_i32())
        r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
                   fn.params[0], fn.params[1], fn.params[2],
                   result_ty=_i32())
        b.ret(r)
        b.end_function()
        return mod
    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_arena_pair_helper_text_pinned():
    """The PAIR helper's parity-sensitive lines (sext width, GEP
    element type, ±1/±2 arithmetic) are pinned by exact substring —
    mirrors the existing arena_push pin (audit-fix LOW)."""
    helper = llvm_ir._HELIX_ARENA_PUSH_PAIR_HELPER
    assert "%cursor_plus_one = add i32 %cursor, 1" in helper, helper
    assert "%cursor_plus_two = add i32 %cursor, 2" in helper, helper
    assert ("%left_idx_i64 = sext i32 %cursor_plus_one to i64"
            in helper), helper
    assert ("%right_idx_i64 = sext i32 %cursor_plus_two to i64"
            in helper), helper
    assert ("getelementptr inbounds i32, ptr @__helix_arena_base, "
            "i64 %left_idx_i64") in helper, helper
    assert ("getelementptr inbounds i32, ptr @__helix_arena_base, "
            "i64 %right_idx_i64") in helper, helper
    # Exactly one data store per operand + one cursor write.
    assert helper.count("store i32 %left,") == 1, helper
    assert helper.count("store i32 %right,") == 1, helper


def test_stage206r_arena_triple_helper_text_pinned():
    """Same shape pinning for TRIPLE (third operand uses
    cursor+3)."""
    helper = llvm_ir._HELIX_ARENA_PUSH_TRIPLE_HELPER
    assert "%cursor_plus_one = add i32 %cursor, 1" in helper, helper
    assert "%cursor_plus_two = add i32 %cursor, 2" in helper, helper
    assert "%cursor_plus_three = add i32 %cursor, 3" in helper, helper
    for name in ("left_idx_i64", "middle_idx_i64", "right_idx_i64"):
        assert (f"%{name} = sext i32 %cursor_plus_"
                in helper), (name, helper)
        assert (f"getelementptr inbounds i32, ptr @__helix_arena_base, "
                f"i64 %{name}") in helper, (name, helper)
    # Exactly one data store per operand.
    assert helper.count("store i32 %left,") == 1, helper
    assert helper.count("store i32 %middle,") == 1, helper
    assert helper.count("store i32 %right,") == 1, helper


def test_stage206r_arena_pair_helper_has_three_blocks():
    """Same structural assertion as arena_push has (audit-fix LOW)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("a", _i32()), ("b", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    for label in ("entry:", "in_bounds:", "exit:"):
        assert f"\n{label}\n" in ll, (label, ll)


def test_stage206r_arena_triple_helper_has_three_blocks():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("a", _i32()), ("b", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    for label in ("entry:", "in_bounds:", "exit:"):
        assert f"\n{label}\n" in ll, (label, ll)


def test_stage206r_arena_pair_helper_emitted_once_per_module():
    """N PAIR ops in M functions still emit ONE helper + ONE global."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("a", [("x", _i32()), ("y", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
           result_ty=_i32())
    r1 = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn.params[0], fn.params[1],
                result_ty=_i32())
    b.ret(r1)
    b.end_function()
    fn2 = b.begin_function("c", [("x", _i32()), ("y", _i32())], _i32())
    r2 = b.emit(tir.OpKind.ARENA_PUSH_PAIR, fn2.params[0], fn2.params[1],
                result_ty=_i32())
    b.ret(r2)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("define internal i32 @__helix_arena_push_pair(") == 1, ll
    assert ll.count("@__helix_arena_base = internal global") == 1, ll
    assert ll.count("call i32 @__helix_arena_push_pair(") == 3, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_arena_triple_helper_emitted_once_per_module():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "a", [("x", _i32()), ("y", _i32()), ("z", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
           fn.params[0], fn.params[1], fn.params[2], result_ty=_i32())
    r = b.emit(tir.OpKind.ARENA_PUSH_TRIPLE,
               fn.params[0], fn.params[1], fn.params[2], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count(
        "define internal i32 @__helix_arena_push_triple(") == 1, ll
    assert ll.count("call i32 @__helix_arena_push_triple(") == 2, ll
    assert llvm_ir.mock_validate_ll(ll) == []


# ==========================================================================
# Stage 206-R chunk — TRACE_ENTRY / TRACE_EXIT
# First void-returning helper (`@__helix_trace_event(i32 fn_id, i32 kind)
# -> void`) + two new module globals (`@__helix_trace_count: i32`,
# `@__helix_trace_buf: [2*CAP x i32]`). emit_module pre-pass interns
# fn_names from every TRACE op in module-walk order; the resulting
# stable name->id table is shared across every _FnEmitter so fn_ids
# are consistent across functions.
# ==========================================================================
def test_stage206r_trace_cap_matches_x86_backend():
    """`_HELIX_TRACE_CAP` MUST equal `x86_64.HELIX_TRACE_CAP` —
    Stage 207 parity gate compares the buffer's overflow point
    across both backends.

    v3.1 step 6a: tautological since both sides import from
    `_shared_constants`. The source-grep drift guard lives in
    `test_shared_constants.py`."""
    from helixc.backend import x86_64
    assert llvm_ir._HELIX_TRACE_CAP == x86_64.HELIX_TRACE_CAP


def test_stage206r_emit_trace_entry_exit_calls():
    """TRACE_ENTRY emits `call void @__helix_trace_event(i32 fn_id,
    i32 0)`; TRACE_EXIT emits the same with `i32 1`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "main"})
    r = b.const_int(0)
    b.emit(tir.OpKind.TRACE_EXIT, r, attrs={"fn_name": "main"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # fn_id = 0 (first and only interned fn_name).
    assert "call void @__helix_trace_event(i32 0, i32 0)" in ll, ll
    assert "call void @__helix_trace_event(i32 0, i32 1)" in ll, ll
    # Helper + both globals emitted exactly once.
    assert ll.count("define internal void @__helix_trace_event(") == 1, ll
    assert ll.count("@__helix_trace_count = internal global") == 1, ll
    assert ll.count("@__helix_trace_buf = internal global") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_trace_globals_not_emitted_when_unused():
    """A module with no TRACE ops must NOT emit the trace globals
    OR the helper — same lazy-emission discipline as the arena
    global."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.ret(b.const_int(1))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "@__helix_trace_count" not in ll, ll
    assert "@__helix_trace_buf" not in ll, ll
    assert "@__helix_trace_event" not in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_intern_trace_fn_ids_deterministic_order():
    """Pre-pass interns fn_names in module-walk order: function
    insertion order, then block order, then op order. Same module
    -> same id table -> same emitted constants."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("alpha", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "alpha"})
    b.ret(b.const_int(0))
    b.end_function()
    b.begin_function("beta", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "beta"})
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "alpha"})  # repeat
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "gamma"})
    b.ret(b.const_int(0))
    b.end_function()
    fn_ids = llvm_ir._intern_trace_fn_ids(mod)
    # Insertion order: alpha first (from `alpha`'s body), beta next
    # (from `beta`'s first op), gamma last (from `beta`'s third op
    # -- the second op is a repeat of alpha, so no new id).
    assert fn_ids == {"alpha": 0, "beta": 1, "gamma": 2}


def test_stage206r_intern_trace_fn_ids_shared_across_emitters():
    """A fn_name appearing in TRACE ops in two different functions
    resolves to the SAME id — the interning table is shared by
    every _FnEmitter in emit_module."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("a", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "shared"})
    b.ret(b.const_int(0))
    b.end_function()
    b.begin_function("c", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "shared"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # Both functions reference fn_id 0 (the only interned name).
    assert ll.count("call void @__helix_trace_event(i32 0, i32 0)") == 2, ll
    # No other fn_id is used.
    assert "i32 1, i32 0)" not in ll, ll


def test_stage206r_intern_trace_fn_ids_skips_extern():
    """`_intern_trace_fn_ids` skips `is_extern` functions (they
    have no body to walk)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("extern_fn", [], _i32(),
                     attrs={"is_extern": True})
    b.end_function()
    b.begin_function("main", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "main"})
    b.ret(b.const_int(0))
    b.end_function()
    fn_ids = llvm_ir._intern_trace_fn_ids(mod)
    # "extern_fn" is not in the table even though it's a function
    # name — its body wasn't scanned. "main" is interned because
    # its body has a TRACE_ENTRY.
    assert fn_ids == {"main": 0}


def test_stage206r_intern_trace_fn_ids_skips_non_trace_ops():
    """The pre-pass walks every op but only interns TRACE_ENTRY /
    TRACE_EXIT — unrelated ops don't pollute the table."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    fn_ids = llvm_ir._intern_trace_fn_ids(mod)
    assert fn_ids == {"f": 0}


def test_stage206r_intern_trace_fn_ids_skips_malformed_fn_name():
    """A malformed `fn_name` attr is left for the per-op handler to
    reject (the pre-pass intentionally stays total over the
    module). Test confirms a missing `fn_name` doesn't crash
    `_intern_trace_fn_ids`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY)  # no fn_name attr
    b.ret(b.const_int(0))
    b.end_function()
    fn_ids = llvm_ir._intern_trace_fn_ids(mod)
    assert fn_ids == {}


def test_stage206r_trace_event_helper_has_three_blocks():
    """The trace_event helper has three labelled blocks (entry /
    store / skip)."""
    helper = llvm_ir._HELIX_TRACE_EVENT_HELPER
    for label in ("entry:", "store:", "skip:"):
        assert f"\n{label}\n" in helper, (label, helper)


def test_stage206r_trace_event_helper_text_pinned():
    """Parity-sensitive lines of the trace_event helper. A mutation
    of any one (full-buffer predicate, shift width, GEP element
    type, store ordering) would silently produce a wrong layout
    that x86 cannot read back."""
    helper = llvm_ir._HELIX_TRACE_EVENT_HELPER
    assert (f"icmp uge i32 %count, {llvm_ir._HELIX_TRACE_CAP}"
            in helper), helper
    # Each event is 2 i32s, so fn_id index = count * 2; the shl by
    # 1 is the equivalent of count << 1 = count * 2.
    assert "%fn_id_idx = shl i64 %count_i64, 1" in helper, helper
    assert "%kind_idx = add i64 %fn_id_idx, 1" in helper, helper
    assert ("getelementptr inbounds i32, ptr @__helix_trace_buf, "
            "i64 %fn_id_idx") in helper, helper
    assert ("getelementptr inbounds i32, ptr @__helix_trace_buf, "
            "i64 %kind_idx") in helper, helper
    # cursor advance must store cursor+1 back to the SAME slot we
    # loaded from.
    assert "%new_count = add i32 %count, 1" in helper, helper
    assert ("store i32 %new_count, ptr @__helix_trace_count"
            in helper), helper
    # void return.
    assert "ret void" in helper, helper


def test_stage206r_trace_event_helper_atomic_on_overflow():
    """No `store i32` runs in entry / skip blocks (only in
    store). On full buffer, the helper branches `entry -> skip`
    directly and returns void with no side effects."""
    helper = llvm_ir._HELIX_TRACE_EVENT_HELPER
    assert ("br i1 %full, label %skip, label %store" in helper), helper
    entry_block = helper.split("store:")[0]
    assert "store i32" not in entry_block, entry_block
    skip_block = helper.split("skip:")[1]
    assert "store i32" not in skip_block, skip_block


def test_stage206r_trace_buf_size_matches_event_layout():
    """The buffer global is `[2 * CAP x i32]` so each event (i32
    fn_id + i32 kind) fits in two i32 slots — 8 bytes total per
    event, matching x86_64.py's `HELIX_TRACE_CAP * 8` byte layout."""
    cap = llvm_ir._HELIX_TRACE_CAP
    assert (f"@__helix_trace_buf = internal global "
            f"[{cap * 2} x i32] zeroinitializer"
            in llvm_ir._HELIX_TRACE_BUF_GLOBAL_DEF), (
        llvm_ir._HELIX_TRACE_BUF_GLOBAL_DEF)


def test_stage206r_trace_count_starts_at_zero():
    """The cursor global zeros at module load — the first event
    lands at slot 0 (matches x86 which `zeroinitializer`s BSS)."""
    assert ("@__helix_trace_count = internal global i32 0"
            in llvm_ir._HELIX_TRACE_COUNT_GLOBAL_DEF), (
        llvm_ir._HELIX_TRACE_COUNT_GLOBAL_DEF)


def test_stage206r_trace_event_helper_emitted_once_per_module():
    """Many TRACE ops in many functions still emit ONE helper and
    ONE set of globals."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("a", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "a"})
    b.emit(tir.OpKind.TRACE_EXIT, attrs={"fn_name": "a"})
    b.ret(b.const_int(0))
    b.end_function()
    b.begin_function("c", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "c"})
    b.emit(tir.OpKind.TRACE_EXIT, attrs={"fn_name": "c"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("define internal void @__helix_trace_event(") == 1, ll
    assert ll.count("@__helix_trace_count = internal global") == 1, ll
    assert ll.count("@__helix_trace_buf = internal global") == 1, ll
    assert ll.count("call void @__helix_trace_event(") == 4, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_trace_is_deterministic():
    """Two emits of the same module produce byte-identical output
    (deterministic pre-pass + sorted-by-name global / helper
    emission)."""
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("alpha", [], _i32())
        b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "alpha"})
        b.ret(b.const_int(0))
        b.end_function()
        b.begin_function("beta", [], _i32())
        b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "beta"})
        b.ret(b.const_int(0))
        b.end_function()
        return mod
    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_trace_entry_rejects_operands():
    """TRACE_ENTRY takes NO operands — fail closed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, b.const_int(0),
           attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="TRACE_ENTRY takes no operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_trace_exit_accepts_zero_or_one_operands():
    """TRACE_EXIT optionally takes one operand (the return value
    for liveness). Both forms must succeed."""
    # Zero operands.
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_EXIT, attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "call void @__helix_trace_event(i32 0, i32 1)" in ll, ll

    # One operand.
    mod2 = tir.Module()
    b2 = tir.IRBuilder(mod2)
    b2.begin_function("f", [], _i32())
    v = b2.const_int(42)
    b2.emit(tir.OpKind.TRACE_EXIT, v, attrs={"fn_name": "f"})
    b2.ret(v)
    b2.end_function()
    ll2 = llvm_ir.emit_module(mod2)
    assert "call void @__helix_trace_event(i32 0, i32 1)" in ll2, ll2


def test_stage206r_trace_exit_rejects_two_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_EXIT, b.const_int(0), b.const_int(1),
           attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="TRACE_EXIT expects zero or one operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_trace_rejects_op_with_result():
    """TRACE_ENTRY / TRACE_EXIT are VOID — having a result is
    malformed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, result_ty=_i32(),
           attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="has no result"):
        llvm_ir.emit_module(mod)


def test_stage206r_trace_rejects_missing_fn_name_attr():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY)  # no fn_name
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="non-empty 'fn_name' string attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_trace_rejects_empty_fn_name():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": ""})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="non-empty 'fn_name' string attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_trace_rejects_non_string_fn_name():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": 42})
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="non-empty 'fn_name' string attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_trace_via_emit_function_fails_closed():
    """A TRACE op handed to `_FnEmitter` directly (without a
    module-level pre-pass) has no fn-id table to resolve against
    -- fail closed with a clear diagnostic pointing the caller at
    `emit_module`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    fn = next(iter(mod.functions.values()))
    # Construct without trace_fn_ids; calling .emit() should fail.
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="emit this module via `emit_module"):
        llvm_ir._FnEmitter(fn).emit()


def test_stage206r_trace_exit_keepalive_bitcast_emitted_for_operand():
    """TRACE_EXIT with an operand emits a `bitcast` that forces an
    LLVM-IR use of the operand — mirrors x86's `mov eax, [slot]`
    load that keeps the value alive past the trace call.
    Audit-fix MEDIUM-2 (silent-failure)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [], _i32())
    v = b.const_int(42)
    b.emit(tir.OpKind.TRACE_EXIT, v, attrs={"fn_name": "f"})
    b.ret(v)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # Operand of TRACE_EXIT here is the const_int 42 -> inline
    # literal `42`. Keepalive bitcast must reference it.
    assert "%trace_keepalive.0 = bitcast i32 42 to i32" in ll, ll


def test_stage206r_trace_exit_keepalive_skipped_when_no_operand():
    """TRACE_EXIT with no operand has no value to keep alive — no
    keepalive bitcast is emitted."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_EXIT, attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "%trace_keepalive" not in ll, ll


def test_stage206r_trace_exit_keepalive_index_increments_per_op():
    """Multiple TRACE_EXITs with operands in the same function get
    distinct keepalive register names so they don't clash."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("v", _i32())], _i32())
    b.emit(tir.OpKind.TRACE_EXIT, fn.params[0], attrs={"fn_name": "f"})
    b.emit(tir.OpKind.TRACE_EXIT, fn.params[0], attrs={"fn_name": "f"})
    b.emit(tir.OpKind.TRACE_EXIT, fn.params[0], attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    for idx in (0, 1, 2):
        assert (f"%trace_keepalive.{idx} = bitcast i32 %v0 to i32"
                in ll), (idx, ll)


def test_stage206r_intern_trace_fn_ids_skips_kernel():
    """`_intern_trace_fn_ids` skips `@kernel` functions for the
    same reason it skips `is_extern`: kernels are rejected by
    `emit_module` so their TRACE ops never produce IR — latent
    defense against any future relaxation of the kernel rejection.
    Audit-fix LOW (code-reviewer / type-design)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("kern", [], _i32(), attrs={"kernel": True})
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "kern_inner"})
    b.ret(b.const_int(0))
    b.end_function()
    b.begin_function("host", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "host"})
    b.ret(b.const_int(0))
    b.end_function()
    fn_ids = llvm_ir._intern_trace_fn_ids(mod)
    # `kern_inner` is NOT interned even though it appeared first —
    # the kernel function is skipped, so `host` gets id 0.
    assert "kern_inner" not in fn_ids
    assert fn_ids == {"host": 0}


def test_stage206r_trace_diagnostic_names_both_root_causes():
    """The "fn_id not in table" diagnostic must name BOTH possible
    causes (hand-built dict vs concurrent mutation) so a developer
    sees the more-likely cause first. Audit-fix MEDIUM-3
    (silent-failure)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.TRACE_ENTRY, attrs={"fn_name": "f"})
    b.ret(b.const_int(0))
    b.end_function()
    fn = next(iter(mod.functions.values()))
    # Construct with a hand-built dict that omits the fn_name.
    emitter = llvm_ir._FnEmitter(fn, trace_fn_ids={"other_name": 0})
    with pytest.raises(llvm_ir.LLVMEmitError) as exc_info:
        emitter.emit()
    msg = str(exc_info.value)
    assert "hand-built" in msg, msg
    assert "concurrent mutation" in msg, msg
    assert "emit_module" in msg, msg


def test_stage206r_helper_spec_advertises_ret_ty():
    """Every helper in the registry advertises its return type
    explicitly via `_HelperFunctionSpec.ret_ty`. Cross-checked at
    module load by `_check_helper_function_table`. Audit-fix
    type-design polish #1 (NOW that void helpers exist)."""
    assert (llvm_ir._HELPER_FUNCTIONS["__helix_trace_event"].ret_ty
            == "void")
    for arena_name in ("__helix_arena_push", "__helix_arena_get",
                       "__helix_arena_set", "__helix_arena_len",
                       "__helix_arena_push_pair",
                       "__helix_arena_push_triple"):
        assert (llvm_ir._HELPER_FUNCTIONS[arena_name].ret_ty == "i32"), (
            arena_name)
    assert (llvm_ir._HELPER_FUNCTIONS["__helix_print_int"].ret_ty
            == "i32")


def test_stage206r_helper_spec_ret_ty_drift_rejected():
    """A `_HelperFunctionSpec` whose `ret_ty` does not match the
    helper's `define internal <ret_ty>` line is rejected at
    construction (call-site signature drift would otherwise emit
    invalid LLVM IR that `mock_validate_ll` does not detect)."""
    with pytest.raises(
            ValueError, match="ret_ty 'i64' does not match"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ret_ty="i64",  # claims i64, but body says i32
            ffi_declares=(),
        )


def test_stage206r_helper_spec_rejects_empty_ret_ty():
    with pytest.raises(ValueError, match="ret_ty must be"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ret_ty="",
            ffi_declares=(),
        )


def test_stage206r_print_kind_catchall_does_not_mention_landed_ops():
    """The PRINT _kind catchall must not advertise already-landed
    ops as "later chunks" — audit-fix MEDIUM-1 (silent-failure /
    documentation drift)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "unknown_kind"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError) as exc_info:
        llvm_ir.emit_module(mod)
    msg = str(exc_info.value)
    # The landed ops must not appear as residual.
    assert "TRACE_ENTRY/EXIT" not in msg, msg
    # The actually-pending residual must.
    assert "read_file_to_arena" in msg, msg


# ==========================================================================
# Stage 206-R chunk — PRINT.read_file_to_arena
# Opens path O_RDONLY into 1MB stack buffer; traps via @llvm.trap on
# truncation (read == BUF_SIZE sentinel mirrors x86's ud2); pushes
# each byte to the arena via __helix_arena_push. New
# `_HelperFunctionSpec.helper_deps` field for transitive helper deps
# (read_file_to_arena depends on arena_push).
# ==========================================================================
def test_stage206r_read_file_buf_size_matches_x86_backend():
    """`_HELIX_READ_FILE_BUF_SIZE` (1 MiB) must match
    x86_64.py's BUF_SIZE in `read_file_to_arena` so both backends
    trap on the same input file size."""
    assert llvm_ir._HELIX_READ_FILE_BUF_SIZE == 0x100000


def test_stage206r_emit_read_file_to_arena_call():
    """A read_file_to_arena PRINT lowers to a single call to
    `@__helix_read_file_to_arena(ptr <path>)` — the helper handles
    everything (open / read / close / trap / per-byte arena push)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/src.hx"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # Path string is NUL-terminated.
    assert "[12 x i8] c\"/tmp/src.hx\\00\"" in ll, ll
    # Single call site.
    assert (f"%v{r.id} = call i32 @__helix_read_file_to_arena(ptr "
            in ll), ll
    # Helper defined once; transitive arena_push helper + arena
    # global also pulled in.
    assert ll.count(
        "define internal i32 @__helix_read_file_to_arena(") == 1, ll
    assert ll.count(
        "define internal i32 @__helix_arena_push(") == 1, ll
    assert "@__helix_arena_base = internal global" in ll, ll
    # All four FFI declares present.
    assert "declare i32 @open(ptr, i32, i32)" in ll, ll
    assert "declare i64 @read(i32, ptr, i64)" in ll, ll
    assert "declare i32 @close(i32)" in ll, ll
    assert "declare void @llvm.trap()" in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_read_file_helper_text_pinned():
    """Parity-sensitive lines: open args (O_RDONLY=0, mode=0), the
    truncation-sentinel predicate (nread == BUF_SIZE), trap+
    unreachable, sign-clamp pattern, loop-header phi, per-byte
    zext/push call."""
    helper = llvm_ir._HELIX_READ_FILE_TO_ARENA_HELPER
    # open(path, O_RDONLY=0, mode=0) — matches x86 line 3464-3466.
    assert "call i32 @open(ptr %path, i32 0, i32 0)" in helper, helper
    # read(fd, buf, BUF_SIZE) — full buffer size as i64.
    bufsize = llvm_ir._HELIX_READ_FILE_BUF_SIZE
    assert (f"call i64 @read(i32 %fd, ptr %buf, i64 {bufsize})"
            in helper), helper
    # Truncation sentinel.
    assert (f"icmp eq i64 %nread, {bufsize}" in helper), helper
    assert "call void @llvm.trap()" in helper, helper
    assert "unreachable" in helper, helper
    # Sign-clamp: select on icmp slt.
    assert "%is_neg = icmp slt i32 %nread_i32, 0" in helper, helper
    assert ("%nread_clamped = select i1 %is_neg, i32 0, i32 %nread_i32"
            in helper), helper
    # Loop-header phi sets i=0 from sign_check, i+1 from loop_body.
    assert ("%i = phi i32 [ 0, %sign_check ], [ %i_next, %loop_body ]"
            in helper), helper
    # Per-byte push.
    assert ("%byte_i32 = zext i8 %byte to i32" in helper), helper
    assert ("%push_ret = call i32 @__helix_arena_push(i32 %byte_i32)"
            in helper), helper


def test_stage206r_read_file_helper_has_six_blocks():
    """The read_file_to_arena helper has six labelled blocks
    (entry / trap / sign_check / loop_header / loop_body / exit)."""
    helper = llvm_ir._HELIX_READ_FILE_TO_ARENA_HELPER
    for label in ("entry:", "trap:", "sign_check:",
                  "loop_header:", "loop_body:", "exit:"):
        assert f"\n{label}\n" in helper, (label, helper)


def test_stage206r_read_file_traps_on_truncation_via_llvm_trap():
    """The truncation branch calls `@llvm.trap()` (not `exit`) so
    the process dies via SIGILL — matches x86's literal `ud2`."""
    helper = llvm_ir._HELIX_READ_FILE_TO_ARENA_HELPER
    # The trap block contains EXACTLY one llvm.trap call followed
    # by unreachable, in that order.
    trap_block = helper.split("trap:")[1].split("sign_check:")[0]
    assert "call void @llvm.trap()" in trap_block, trap_block
    assert "unreachable" in trap_block, trap_block


def test_stage206r_read_file_loop_push_result_is_discarded():
    """The per-byte push call's return value is bound (`%push_ret`)
    but never used — matches x86's "loop counter advances regardless
    of arena_push success" semantics (line 3537). A full arena
    returns -1 from each push but the loop continues."""
    helper = llvm_ir._HELIX_READ_FILE_TO_ARENA_HELPER
    # The bound register exists exactly once (the call site)
    # — it's NEVER referenced elsewhere in the helper body.
    assert helper.count("%push_ret") == 1, helper


def test_stage206r_read_file_rejects_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, b.const_int(0), result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/x"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="read_file_to_arena PRINT takes no "
                             "operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_read_file_rejects_missing_path_attr():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'path' attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_read_file_rejects_non_string_path():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena", "path": 42})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="string 'path' attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_read_file_rejects_embedded_nul_in_path():
    """Same HIGH-1 guard as write_file — open(2) reads a C-string
    and stops at the first NUL; an embedded NUL would silently
    truncate the path."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/a\x00/etc/shadow"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="embedded NUL"):
        llvm_ir.emit_module(mod)


def test_stage206r_read_file_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.PRINT, result_ty=tir.TIRScalar("i64"),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/x"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="PRINT yields an i32"):
        llvm_ir.emit_module(mod)


def test_stage206r_read_file_to_arena_is_deterministic():
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("main", [], _i32())
        r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
                   attrs={"_kind": "read_file_to_arena",
                          "path": "/tmp/src.hx"})
        b.ret(r)
        b.end_function()
        return mod
    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_read_file_transitive_dep_pulls_arena_push():
    """`__helix_read_file_to_arena` declares
    `helper_deps=("__helix_arena_push",)` — registering the read
    helper must pull in arena_push (and its module-global) even when
    no ARENA_PUSH op appears in the source module."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/x"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # No explicit ARENA_PUSH op, but the helper + global must be
    # present because read_file_to_arena depends on them.
    assert "define internal i32 @__helix_arena_push(" in ll, ll
    assert "@__helix_arena_base = internal global" in ll, ll


def test_stage206r_read_file_dedups_with_explicit_arena_push():
    """If a module uses BOTH read_file_to_arena AND a direct
    ARENA_PUSH op, the arena_push helper is still emitted exactly
    once (the recursive registration is idempotent via
    `name in self.helper_functions`)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("main", [("v", _i32())], _i32())
    b.emit(tir.OpKind.ARENA_PUSH, fn.params[0], result_ty=_i32())
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/x"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count(
        "define internal i32 @__helix_arena_push(") == 1, ll
    assert ll.count(
        "define internal i32 @__helix_read_file_to_arena(") == 1, ll
    assert ll.count("@__helix_arena_base = internal global") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_read_file_helper_spec_advertises_helper_deps():
    """The registry must record the helper-dep so the drift guard
    cross-checks it against the helper body."""
    spec = llvm_ir._HELPER_FUNCTIONS["__helix_read_file_to_arena"]
    assert spec.helper_deps == ("__helix_arena_push",)


def test_stage206r_helper_spec_rejects_unknown_helper_dep():
    """A helper_deps entry must resolve in
    `_HELPER_FUNCTIONS_AUTHORITY` — the drift guard catches typos
    at module load."""
    original = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_bad_dep"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_bad_dep() {\n"
                    "entry:\n"
                    "  %x = call i32 @__helix_nonexistent()\n"
                    "  ret i32 %x\n"
                    "}"),
                ret_ty="i32",
                ffi_declares=(),
                helper_deps=("__helix_nonexistent",),
            ))
        with pytest.raises(AssertionError,
                           match="declares helper-dep on"):
            llvm_ir._check_helper_function_table()
    finally:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original)


def test_stage206r_helper_spec_rejects_helper_deps_body_drift():
    """A helper_deps entry whose name is not called in the helper
    body has drifted — the drift guard catches it (analogous to the
    ffi_declares cross-check)."""
    original = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_no_call_drift"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_no_call_drift() {\n"
                    "entry:\n"
                    "  ret i32 0\n"
                    "}"),
                ret_ty="i32",
                ffi_declares=(),
                helper_deps=("__helix_arena_push",),
            ))
        with pytest.raises(AssertionError,
                           match="registry and body have drifted"):
            llvm_ir._check_helper_function_table()
    finally:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original)


def test_stage206r_helper_spec_rejects_malformed_helper_deps():
    with pytest.raises(ValueError, match="helper_deps must be"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ret_ty="i32",
            ffi_declares=(),
            helper_deps="not_a_tuple",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="helper_deps entry"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ret_ty="i32",
            ffi_declares=(),
            helper_deps=("user_helper",),
        )


def test_stage206r_helper_spec_rejects_helper_deps_duplicates():
    with pytest.raises(ValueError,
                       match="helper_deps has duplicates"):
        llvm_ir._HelperFunctionSpec(
            definition=(
                "define internal i32 @__helix_x() {\n"
                "entry:\n"
                "  ret i32 0\n"
                "}"),
            ret_ty="i32",
            ffi_declares=(),
            helper_deps=("__helix_arena_push", "__helix_arena_push"),
        )


def test_stage206r_helper_deps_cycle_detected_at_module_load():
    """Audit-fix HIGH-1: a true `helper_deps` cycle would otherwise
    leak as a raw `RecursionError` (the early-return idempotency
    check sets the visited marker AFTER the recursive walk). The
    drift guard's DFS-based cycle detector fires at module load
    with an actionable diagnostic naming the cycle."""
    original = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    try:
        # Construct a 2-node cycle A <-> B.
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_cyc_a"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_cyc_a() {\n"
                    "entry:\n"
                    "  %x = call i32 @__helix_cyc_b()\n"
                    "  ret i32 %x\n"
                    "}"),
                ret_ty="i32",
                ffi_declares=(),
                helper_deps=("__helix_cyc_b",),
            ))
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_cyc_b"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_cyc_b() {\n"
                    "entry:\n"
                    "  %y = call i32 @__helix_cyc_a()\n"
                    "  ret i32 %y\n"
                    "}"),
                ret_ty="i32",
                ffi_declares=(),
                helper_deps=("__helix_cyc_a",),
            ))
        with pytest.raises(AssertionError, match="cycle detected"):
            llvm_ir._check_helper_function_table()
    finally:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original)


def test_stage206r_helper_deps_self_cycle_detected():
    """Same audit fix: a self-cycle (A depends on A) is the simplest
    cycle and must also be caught."""
    original = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    try:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_self_cyc"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_self_cyc() {\n"
                    "entry:\n"
                    "  %x = call i32 @__helix_self_cyc()\n"
                    "  ret i32 %x\n"
                    "}"),
                ret_ty="i32",
                ffi_declares=(),
                helper_deps=("__helix_self_cyc",),
            ))
        with pytest.raises(AssertionError, match="cycle detected"):
            llvm_ir._check_helper_function_table()
    finally:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original)


def test_stage206r_drift_guard_ignores_comment_lines():
    """Audit-fix HIGH-2: a `;`-comment-only mention of a call
    pattern must NOT satisfy the body-vs-registry drift check.
    Without `_strip_llvm_comment`, a future helper whose body got
    refactored to remove the real call but retain a comment about
    it would silently pass the drift guard."""
    original = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    try:
        # Comment-only mention of @__helix_arena_push — no real call.
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_comment_only"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_comment_only() {\n"
                    "entry:\n"
                    "  ; the real arena_push call: "
                    "call i32 @__helix_arena_push(i32 0)\n"
                    "  ret i32 0\n"
                    "}"),
                ret_ty="i32",
                ffi_declares=(),
                helper_deps=("__helix_arena_push",),
            ))
        with pytest.raises(AssertionError,
                           match="registry and body have drifted"):
            llvm_ir._check_helper_function_table()
    finally:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original)


def test_stage206r_drift_guard_strips_trailing_comments():
    """Sibling audit fix: a real `call` line with a trailing
    `; comment` is still recognized (the strip is correct, not
    overzealous)."""
    original = dict(llvm_ir._HELPER_FUNCTIONS_AUTHORITY)
    try:
        # Real call followed by a trailing comment.
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY["__helix_real_call"] = (
            llvm_ir._HelperFunctionSpec(
                definition=(
                    "define internal i32 @__helix_real_call() {\n"
                    "entry:\n"
                    "  %x = call i32 @__helix_arena_push(i32 0)  "
                    "; with trailing comment\n"
                    "  ret i32 %x\n"
                    "}"),
                ret_ty="i32",
                ffi_declares=(),
                helper_deps=("__helix_arena_push",),
            ))
        # Should NOT raise — the real call is detected.
        llvm_ir._check_helper_function_table()
    finally:
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.clear()
        llvm_ir._HELPER_FUNCTIONS_AUTHORITY.update(original)


def test_stage206r_supported_print_kinds_pinned():
    """`_SUPPORTED_PRINT_KINDS` is the single source of truth for
    the PRINT _kind whitelist (audit-fix polish Q4). A new sub-kind
    landing in a future chunk updates ONE constant rather than
    drifting between the dispatch check and the error message."""
    assert llvm_ir._SUPPORTED_PRINT_KINDS == frozenset({
        "print_str", "print_int", "write_file", "read_file_to_arena",
    })


def test_stage206r_read_file_truncation_predicate_is_eq_not_relaxed():
    """Audit-fix MEDIUM-3: pin that the truncation sentinel uses
    exactly `icmp eq i64 %nread, BUF_SIZE` — a relaxed comparator
    (uge / sgt / ne) would either silently widen the trap or
    silently let truncation through."""
    helper = llvm_ir._HELIX_READ_FILE_TO_ARENA_HELPER
    bufsize = llvm_ir._HELIX_READ_FILE_BUF_SIZE
    # Find the i64 icmp on %nread (NOT the post-trunc i32 sign
    # check on `%nread_i32`, NOT the loop's i32 done check on
    # `%nread_clamped`).
    nread_i64_cmps = [
        ln.strip() for ln in helper.splitlines()
        if "icmp" in ln and "i64 %nread," in ln
    ]
    assert len(nread_i64_cmps) == 1, nread_i64_cmps
    assert (f"icmp eq i64 %nread, {bufsize}"
            in nread_i64_cmps[0]), nread_i64_cmps


def test_stage206r_read_file_dedups_same_path():
    """Two read_file_to_arena PRINTs with the same path share one
    string global (content-addressed via SHA-256). Parallel to the
    existing write_file dedup test (audit-fix LOW-1)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(),
           attrs={"_kind": "read_file_to_arena", "path": "/tmp/x"})
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/x"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # Exactly one path constant (the helper's own globals are
    # @__helix_arena_base, not strings).
    assert ll.count('private unnamed_addr constant [') == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_read_file_distinct_paths_distinct_globals():
    """Two read_file_to_arena PRINTs with DIFFERENT paths each get
    their own string global."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.emit(tir.OpKind.PRINT, result_ty=_i32(),
           attrs={"_kind": "read_file_to_arena",
                  "path": "/tmp/one"})
    r = b.emit(tir.OpKind.PRINT, result_ty=_i32(),
               attrs={"_kind": "read_file_to_arena",
                      "path": "/tmp/two"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count('private unnamed_addr constant [') == 2, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_register_helper_function_is_idempotent():
    """Calling `_register_helper_function` twice for the same name
    is a no-op — the early `if name in self.helper_functions:
    return` short-circuit prevents both double-registration and
    infinite recursion on cycles."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    fn = next(iter(mod.functions.values()))
    emitter = llvm_ir._FnEmitter(fn)
    emitter._register_helper_function("__helix_arena_push")
    emitter._register_helper_function("__helix_arena_push")
    emitter._register_helper_function("__helix_arena_push")
    assert emitter.helper_functions == {"__helix_arena_push"}
    # The module-global was added exactly once.
    assert emitter.module_globals == {"__helix_arena_base"}


# ==========================================================================
# Stage 206-R chunk — QUOTE / SPLICE / MODIFY (+ reflection cells)
# Final 206-R chunk. AGI metaprogramming primitives over a 64-cell
# i64 reflection-state array. QUOTE: pure inline `add i32 0,
# <handle>` (compile-time ast_handle mod NUM_CELLS). SPLICE: 3-block
# helper, bounds-checked load, returns 0 on OOB. MODIFY: 4-block
# helper takes a verifier function pointer, returns 1 on accepted-
# store, 0 on OOB or verifier-reject. REFLECT_HASH unimplemented in
# both backends — lands in the catchall fail-closed.
# ==========================================================================
def test_stage206r_num_cells_matches_x86_backend():
    """`_HELIX_NUM_CELLS` (64) must match
    `x86_64.py::HELIX_NUM_CELLS` — Stage 207 parity gate compares
    both backends against this single overflow point.

    v3.1 step 6a: tautological since both sides import from
    `_shared_constants`. The source-grep drift guard lives in
    `test_shared_constants.py`."""
    from helixc.backend import x86_64
    assert llvm_ir._HELIX_NUM_CELLS == x86_64.HELIX_NUM_CELLS


def test_stage206r_emit_quote_inlines_handle_mod_num_cells():
    """QUOTE materialises `ast_handle % NUM_CELLS` as a pure
    `add i32 0, <handle>` inline emission — matches x86's
    `handle = int(op.attrs.get("ast_handle", 0)) % HELIX_NUM_CELLS`
    at line 4473."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    # ast_handle 100 → 100 % 64 = 36.
    r = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
               attrs={"ast_handle": 100})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = add i32 0, 36" in ll, ll
    # QUOTE does NOT touch the state global; SPLICE/MODIFY do.
    assert "@__helix_state_base" not in ll, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_emit_splice_call():
    """SPLICE lowers to a single `call i32 @__helix_splice(i32 %h)`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32())], _i32())
    r = b.emit(tir.OpKind.SPLICE, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = call i32 @__helix_splice(i32 %v0)" in ll, ll
    assert ll.count("define internal i32 @__helix_splice(") == 1, ll
    assert ll.count("@__helix_state_base = internal global") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_emit_modify_call():
    """MODIFY lowers to `call i32 @__helix_modify(i32 %h, i32 %v,
    ptr @<verifier>)` — the verifier function pointer is hard-coded
    by name at the call site."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("verify", [("h", _i32()), ("v", _i32())], _i32())
    b.ret(b.const_int(1))
    b.end_function()
    fn = b.begin_function("f", [("h", _i32()), ("v", _i32())], _i32())
    h, v = fn.params
    r = b.emit(tir.OpKind.MODIFY, h, v,
               result_ty=_i32(), attrs={"verifier_fn": "verify"})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 @__helix_modify("
            f"i32 %v{h.id}, i32 %v{v.id}, ptr @verify)"
            in ll), ll
    assert ll.count("define internal i32 @__helix_modify(") == 1, ll


def test_stage206r_splice_helper_text_pinned():
    """The splice helper's parity-sensitive lines (OOB predicate,
    sext, GEP element type i64, trunc to i32, phi from entry+load)."""
    helper = llvm_ir._HELIX_SPLICE_HELPER
    cells = llvm_ir._HELIX_NUM_CELLS
    assert "%neg = icmp slt i32 %handle, 0" in helper, helper
    assert f"%big = icmp sge i32 %handle, {cells}" in helper, helper
    assert "%oob = or i1 %neg, %big" in helper, helper
    assert "%handle_i64 = sext i32 %handle to i64" in helper, helper
    assert ("getelementptr inbounds i64, ptr @__helix_state_base, "
            "i64 %handle_i64" in helper), helper
    assert "%loaded = load i64, ptr %slot_ptr, align 8" in helper, helper
    assert "%trunc = trunc i64 %loaded to i32" in helper, helper
    assert ("%result = phi i32 [ 0, %entry ], [ %trunc, %load ]"
            in helper), helper


def test_stage206r_modify_helper_text_pinned():
    """The modify helper's parity-sensitive lines: bounds, verifier
    call through %verifier function pointer, conditional store
    (i32 sext to i64 to match cell width), three-way phi at exit."""
    helper = llvm_ir._HELIX_MODIFY_HELPER
    cells = llvm_ir._HELIX_NUM_CELLS
    assert "%neg = icmp slt i32 %handle, 0" in helper, helper
    assert f"%big = icmp sge i32 %handle, {cells}" in helper, helper
    assert ("%accepted = call i32 %verifier(i32 %handle, "
            "i32 %new_value)" in helper), helper
    assert "%ok = icmp ne i32 %accepted, 0" in helper, helper
    assert "%handle_i64 = sext i32 %handle to i64" in helper, helper
    assert "%value_i64 = sext i32 %new_value to i64" in helper, helper
    assert "store i64 %value_i64, ptr %slot_ptr, align 8" in helper, helper
    # 3-way phi: 0 from entry (OOB), 0 from verify (rejected),
    # 1 from apply (stored).
    assert ("%result = phi i32 [ 0, %entry ], [ 0, %verify ], "
            "[ 1, %apply ]" in helper), helper


def test_stage206r_splice_helper_has_three_blocks():
    helper = llvm_ir._HELIX_SPLICE_HELPER
    for label in ("entry:", "load:", "exit:"):
        assert f"\n{label}\n" in helper, (label, helper)


def test_stage206r_modify_helper_has_four_blocks():
    helper = llvm_ir._HELIX_MODIFY_HELPER
    for label in ("entry:", "verify:", "apply:", "exit:"):
        assert f"\n{label}\n" in helper, (label, helper)


def test_stage206r_splice_oob_returns_zero_no_load():
    """The OOB path branches `entry -> exit` directly; no load
    instruction lives outside the `load:` block. Parallel to the
    arena helpers' atomic-on-overflow invariant."""
    helper = llvm_ir._HELIX_SPLICE_HELPER
    entry_block = helper.split("load:")[0]
    assert "load i64" not in entry_block, entry_block
    exit_block = helper.split("exit:")[1]
    assert "load i64" not in exit_block, exit_block


def test_stage206r_modify_oob_skips_verifier_and_store():
    """OOB path skips BOTH the verifier call AND the store — they
    live only in the `verify:` and `apply:` blocks respectively.
    Without this, an OOB handle could trigger the verifier (e.g. if
    the verifier reads cell state) — UB."""
    helper = llvm_ir._HELIX_MODIFY_HELPER
    entry_block = helper.split("verify:")[0]
    assert "call i32 %verifier" not in entry_block, entry_block
    assert "store i64" not in entry_block, entry_block
    exit_block = helper.split("exit:")[1]
    assert "call i32 %verifier" not in exit_block, exit_block
    assert "store i64" not in exit_block, exit_block


def test_stage206r_state_globals_emitted_once():
    """A module using all three (QUOTE / SPLICE / MODIFY) plus
    multiple call sites emits ONE state global + each helper once."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("v", [("h", _i32()), ("x", _i32())], _i32())
    b.ret(b.const_int(1))
    b.end_function()
    fn = b.begin_function("f", [("v", _i32())], _i32())
    h1 = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
                attrs={"ast_handle": 0})
    h2 = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
                attrs={"ast_handle": 1})
    b.emit(tir.OpKind.SPLICE, h1, result_ty=_i32())
    b.emit(tir.OpKind.SPLICE, h2, result_ty=_i32())
    m = b.emit(tir.OpKind.MODIFY, h1, fn.params[0],
               result_ty=_i32(), attrs={"verifier_fn": "v"})
    b.ret(m)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert ll.count("@__helix_state_base = internal global") == 1, ll
    assert ll.count("define internal i32 @__helix_splice(") == 1, ll
    assert ll.count("define internal i32 @__helix_modify(") == 1, ll
    # 2 inline `add i32 0, N` quotes, 2 splice calls, 1 modify call.
    assert ll.count("add i32 0, ") == 2, ll
    assert ll.count("call i32 @__helix_splice(") == 2, ll
    assert ll.count("call i32 @__helix_modify(") == 1, ll
    assert llvm_ir.mock_validate_ll(ll) == []


def test_stage206r_state_global_not_emitted_when_unused():
    """QUOTE alone does NOT pull in the state global — only SPLICE
    and MODIFY do (QUOTE is a pure inline `add`)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
               attrs={"ast_handle": 0})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert "@__helix_state_base" not in ll, ll
    assert "define internal i32 @__helix_splice(" not in ll, ll
    assert "define internal i32 @__helix_modify(" not in ll, ll


def test_stage206r_quote_rejects_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("x", _i32())], _i32())
    r = b.emit(tir.OpKind.QUOTE, fn.params[0], result_ty=_i32(),
               attrs={"ast_handle": 0})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="QUOTE takes no operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_quote_rejects_non_i32_result():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], tir.TIRScalar("i64"))
    r = b.emit(tir.OpKind.QUOTE, result_ty=tir.TIRScalar("i64"),
               attrs={"ast_handle": 0})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="QUOTE result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_quote_rejects_non_int_ast_handle():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
               attrs={"ast_handle": "not-an-int"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="int 'ast_handle' attr"):
        llvm_ir.emit_module(mod)


def test_stage206r_splice_rejects_zero_operands():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.SPLICE, result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="SPLICE takes one operand"):
        llvm_ir.emit_module(mod)


def test_stage206r_splice_rejects_non_i32_handle():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", tir.TIRScalar("i64"))], _i32())
    r = b.emit(tir.OpKind.SPLICE, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="SPLICE handle has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_splice_rejects_unknown_value_kind():
    """v3.1 step 4: f32 / f64 SPLICE variants are now LOWERED via
    polymorphic helpers. Truly unknown value_kinds (e.g. 'f128',
    'ptx') still reject — the dispatch table is exhaustive over
    {'i32','f32','f64'} and fail-closed on anything else."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32())], _i32())
    r = b.emit(tir.OpKind.SPLICE, fn.params[0], result_ty=_i32(),
               attrs={"value_kind": "f128"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="SPLICE value_kind 'f128' is not supported"):
        llvm_ir.emit_module(mod)


def test_stage206r_modify_canonical_one_operand_rejected():
    """The CANONICAL MODIFY form (with verifier_fn string attr)
    requires exactly 2 operands; 1 with verifier_fn set is
    malformed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32())], _i32())
    r = b.emit(tir.OpKind.MODIFY, fn.params[0], result_ty=_i32(),
               attrs={"verifier_fn": "v"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="canonical MODIFY .*takes two operands"):
        llvm_ir.emit_module(mod)


def test_stage206r_modify_legacy_3op_truthy_check():
    """Audit-fix HIGH-1: MODIFY with no verifier_fn attr and 3
    operands lowers to a runtime truthy check on operand[2] — the
    legacy form x86_64.py supports at line 4538-4548. Without this
    branch, programs using the dynamic-verifier form would compile
    on x86 but fail on LLVM (a real parity divergence)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("h", _i32()), ("v", _i32()), ("c", _i32())], _i32())
    r = b.emit(tir.OpKind.MODIFY,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # Truthy check + zext.
    assert (f"%v{r.id}.is_ne = icmp ne i32 %v{fn.params[2].id}, 0"
            in ll), ll
    assert (f"%v{r.id} = zext i1 %v{r.id}.is_ne to i32"
            in ll), ll
    # No __helix_modify helper pulled in (legacy form is inline).
    assert "define internal i32 @__helix_modify" not in ll, ll
    # No state global either — legacy form doesn't touch cells.
    assert "@__helix_state_base" not in ll, ll


def test_stage206r_modify_legacy_lt_3op_returns_zero():
    """Audit-fix HIGH-1: MODIFY with no verifier_fn and fewer than
    3 operands degrades to `result = 0` (matches x86 line 4549-
    4550)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32()), ("v", _i32())], _i32())
    r = b.emit(tir.OpKind.MODIFY, fn.params[0], fn.params[1],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = add i32 0, 0" in ll, ll
    assert "define internal i32 @__helix_modify" not in ll, ll


def test_stage206r_modify_legacy_rejects_non_i32_operand2():
    """The legacy form's truthy check requires operand[2] to be
    i32 (mirrors x86's `mov eax, [slot]` 32-bit load)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function(
        "f", [("h", _i32()), ("v", _i32()),
              ("c", tir.TIRScalar("i64"))], _i32())
    r = b.emit(tir.OpKind.MODIFY,
               fn.params[0], fn.params[1], fn.params[2],
               result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="legacy verifier operand has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_quote_rejects_bool_ast_handle():
    """Audit-fix MEDIUM-1: a bool ast_handle would silently wrap
    to 0/1 (since `True % 64 == 1`) — reject explicitly with
    `type(...) is int` for consistency with the CONST_INT
    discipline."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
               attrs={"ast_handle": True})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="int 'ast_handle'"):
        llvm_ir.emit_module(mod)


def test_stage206r_quote_negative_handle_wraps_to_valid_cell():
    """A negative ast_handle wraps via Python `%` semantics into
    [0, NUM_CELLS) — matches x86_64.py line 4473's
    `int(...) % HELIX_NUM_CELLS`. Both backends use the same
    Python modulo (non-negative result for non-negative divisor)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    # -5 % 64 == 59
    r = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
               attrs={"ast_handle": -5})
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert f"%v{r.id} = add i32 0, 59" in ll, ll


def test_stage206r_modify_rejects_empty_verifier_fn():
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32()), ("v", _i32())], _i32())
    r = b.emit(tir.OpKind.MODIFY, fn.params[0], fn.params[1],
               result_ty=_i32(), attrs={"verifier_fn": ""})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="non-empty 'verifier_fn'"):
        llvm_ir.emit_module(mod)


def test_stage206r_modify_rejects_unknown_value_kind():
    """Mirror of `test_stage206r_splice_rejects_unknown_value_kind`
    for MODIFY — f32/f64 now lower; unknown kinds still reject."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32()), ("v", _i32())], _i32())
    r = b.emit(tir.OpKind.MODIFY, fn.params[0], fn.params[1],
               result_ty=_i32(),
               attrs={"verifier_fn": "v", "value_kind": "f128"})
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="MODIFY value_kind 'f128' is not supported"):
        llvm_ir.emit_module(mod)


# --------------------------------------------------------------------------
# v3.1 step 4 — f32/f64 polymorphic SPLICE/MODIFY
# Positive-emission tests. These bypass `emit_module` (which still
# rejects float return types at the module-level shape check) by
# driving `_FnEmitter._emit_op` directly with a synthesized FnIR.
# The unit under test is the dispatch-table routing + the helper
# register-on-use side-effect — both observable from a single
# `_emit_op` call.
#
# DESIGN NOTE (audit HIGH-2 followup): the `_FnEmitter._emit_op`
# bypass is LOAD-BEARING-TEMPORARY. A real end-to-end module that
# returns f32/f64 cannot today round-trip through `emit_module`
# (the float-return rejection is intentional until full
# float-arithmetic support is wired). Until that lands, the
# dispatch table's correctness is pinned by THREE complementary
# tests at this level:
#   1. The positive `_emit_op` tests below (per-op smoke).
#   2. `test_stage206r_polymorphic_dispatch_tables_single_source_of_truth`
#      (cross-checks dispatch keys vs. helper registry ret_ty
#      vs. helper definition text — typo on either side surfaces).
#   3. The Stage 207 parity gate (compares x86_64 emission against
#      LLVM emission for the i32 path; f32/f64 parity will land in
#      a follow-up chunk that lifts the float-return restriction).
# When emit_module loses its float-return rejection, replace these
# tests with end-to-end module-level coverage (a SPLICE result
# feeding `fptosi` into an i32 ret, for example).
# --------------------------------------------------------------------------
def _emit_one_op(op: tir.Op, *,
                 params: list[tir.Value],
                 return_ty: tir.TIRType = None) -> tuple[str, set[str]]:
    """Build a minimal FnIR `f(params...) -> return_ty` whose single
    block contains `op`, run `_prepass` to wire param refs, and emit
    just `op`. Returns (emitted_text, helper_functions_registered)."""
    if return_ty is None:
        return_ty = _i32()
    fn = tir.FnIR(
        name="f", params=params, return_ty=return_ty,
        blocks=[tir.Block(id=0, params=[], ops=[op])])
    emitter = llvm_ir._FnEmitter(fn)
    emitter._prepass()
    text = emitter._emit_op(op)
    return text, set(emitter.helper_functions)


def test_stage206r_splice_f32_emits_float_call_to_helper():
    """SPLICE with value_kind='f32' lowers to a `call float
    @__helix_splice_f32(i32 <handle>)` and pulls the f32 helper in."""
    handle = tir.Value(id=0, ty=_i32())
    result = tir.Value(id=1, ty=tir.TIRScalar("f32"))
    op = tir.Op(kind=tir.OpKind.SPLICE,
                operands=[handle], results=[result],
                attrs={"value_kind": "f32"})
    text, helpers = _emit_one_op(op, params=[handle],
                                 return_ty=tir.TIRScalar("f32"))
    assert text == "%v1 = call float @__helix_splice_f32(i32 %v0)", text
    assert helpers == {"__helix_splice_f32"}


def test_stage206r_splice_f64_emits_double_call_to_helper():
    """SPLICE with value_kind='f64' lowers to a `call double
    @__helix_splice_f64(i32 <handle>)` and pulls the f64 helper in."""
    handle = tir.Value(id=0, ty=_i32())
    result = tir.Value(id=1, ty=tir.TIRScalar("f64"))
    op = tir.Op(kind=tir.OpKind.SPLICE,
                operands=[handle], results=[result],
                attrs={"value_kind": "f64"})
    text, helpers = _emit_one_op(op, params=[handle],
                                 return_ty=tir.TIRScalar("f64"))
    assert text == "%v1 = call double @__helix_splice_f64(i32 %v0)", text
    assert helpers == {"__helix_splice_f64"}


def test_stage206r_splice_f32_rejects_mismatched_result_type():
    """value_kind='f32' with an f64 result type is rejected — the
    dispatch table's expected_ty is checked against the result's
    rendered LLVM type so a shape mismatch can't sneak through."""
    handle = tir.Value(id=0, ty=_i32())
    result = tir.Value(id=1, ty=tir.TIRScalar("f64"))  # mismatch
    op = tir.Op(kind=tir.OpKind.SPLICE,
                operands=[handle], results=[result],
                attrs={"value_kind": "f32"})
    fn = tir.FnIR(
        name="f", params=[handle], return_ty=tir.TIRScalar("f64"),
        blocks=[tir.Block(id=0, params=[], ops=[op])])
    emitter = llvm_ir._FnEmitter(fn)
    emitter._prepass()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="SPLICE value_kind 'f32' requires a "
                             "matching f32 result type"):
        emitter._emit_op(op)


def test_stage206r_modify_f32_emits_float_call_to_helper():
    """MODIFY with value_kind='f32' lowers to a `call i32
    @__helix_modify_f32(i32 <h>, float <v>, ptr @verifier)` and
    pulls the f32 helper in. Result is always i32 (the accepted
    flag) regardless of value_kind."""
    handle = tir.Value(id=0, ty=_i32())
    new_val = tir.Value(id=1, ty=tir.TIRScalar("f32"))
    result = tir.Value(id=2, ty=_i32())
    op = tir.Op(kind=tir.OpKind.MODIFY,
                operands=[handle, new_val], results=[result],
                attrs={"verifier_fn": "v", "value_kind": "f32"})
    text, helpers = _emit_one_op(op, params=[handle, new_val])
    assert text == (
        "%v2 = call i32 @__helix_modify_f32"
        "(i32 %v0, float %v1, ptr @v)"), text
    assert helpers == {"__helix_modify_f32"}


def test_stage206r_modify_f64_emits_double_call_to_helper():
    """MODIFY with value_kind='f64' lowers to a `call i32
    @__helix_modify_f64(i32 <h>, double <v>, ptr @verifier)`."""
    handle = tir.Value(id=0, ty=_i32())
    new_val = tir.Value(id=1, ty=tir.TIRScalar("f64"))
    result = tir.Value(id=2, ty=_i32())
    op = tir.Op(kind=tir.OpKind.MODIFY,
                operands=[handle, new_val], results=[result],
                attrs={"verifier_fn": "v", "value_kind": "f64"})
    text, helpers = _emit_one_op(op, params=[handle, new_val])
    assert text == (
        "%v2 = call i32 @__helix_modify_f64"
        "(i32 %v0, double %v1, ptr @v)"), text
    assert helpers == {"__helix_modify_f64"}


def test_stage206r_modify_f32_rejects_mismatched_new_value_type():
    """value_kind='f32' with an f64 new_value is rejected — the
    helper's signature is `(i32, float, ptr)`, so an f64 SSA value
    would be a type-mismatch in the emitted call site."""
    handle = tir.Value(id=0, ty=_i32())
    new_val = tir.Value(id=1, ty=tir.TIRScalar("f64"))  # mismatch
    result = tir.Value(id=2, ty=_i32())
    op = tir.Op(kind=tir.OpKind.MODIFY,
                operands=[handle, new_val], results=[result],
                attrs={"verifier_fn": "v", "value_kind": "f32"})
    fn = tir.FnIR(
        name="f", params=[handle, new_val], return_ty=_i32(),
        blocks=[tir.Block(id=0, params=[], ops=[op])])
    emitter = llvm_ir._FnEmitter(fn)
    emitter._prepass()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="MODIFY value_kind 'f32' requires a "
                             "matching f32 new_value type"):
        emitter._emit_op(op)


def test_stage206r_polymorphic_helpers_registered_with_correct_types():
    """The four new helpers are in `_HELPER_FUNCTIONS` with the
    right `ret_ty` ('float' for splice_f32, 'double' for splice_f64,
    'i32' for both modify variants — modify's accepted-or-not flag
    is always i32). All four share the cell-state global so the
    reflection cells stay one source of truth."""
    splice_f32 = llvm_ir._HELPER_FUNCTIONS["__helix_splice_f32"]
    splice_f64 = llvm_ir._HELPER_FUNCTIONS["__helix_splice_f64"]
    modify_f32 = llvm_ir._HELPER_FUNCTIONS["__helix_modify_f32"]
    modify_f64 = llvm_ir._HELPER_FUNCTIONS["__helix_modify_f64"]
    assert splice_f32.ret_ty == "float", splice_f32
    assert splice_f64.ret_ty == "double", splice_f64
    assert modify_f32.ret_ty == "i32", modify_f32
    assert modify_f64.ret_ty == "i32", modify_f64
    for spec in (splice_f32, splice_f64, modify_f32, modify_f64):
        assert spec.module_globals is llvm_ir._HELIX_STATE_GLOBALS, spec


def test_stage206r_polymorphic_dispatch_tables_single_source_of_truth():
    """v3.1 step 4 audit-fix HIGH-1: `_SPLICE_DISPATCH` and
    `_MODIFY_DISPATCH` are the SOLE source of truth for the polymorphic
    value_kind sets. The op handlers validate against the dispatch
    `.keys()` instead of duplicating a tuple of known values — so
    adding a new value_kind in one place and forgetting the other
    cannot happen.

    Also pins: every dispatch helper_name resolves to a real
    `_HELPER_FUNCTIONS` entry, and the helper's ret_ty matches the
    dispatch table's declared call_ret_ty / new_value_llvm_ty."""
    # SPLICE: keys must equal {"i32","f32","f64"} (the v3.1 step 4
    # contract). If this set grows, the test list below grows with it.
    assert set(llvm_ir._SPLICE_DISPATCH) == {"i32", "f32", "f64"}
    assert set(llvm_ir._MODIFY_DISPATCH) == {"i32", "f32", "f64"}
    # Every helper name in the dispatch table resolves to a registered
    # helper.
    for value_kind, (helper_name, expected_ty, call_ret_ty) in (
            llvm_ir._SPLICE_DISPATCH.items()):
        spec = llvm_ir._HELPER_FUNCTIONS[helper_name]
        assert spec.ret_ty == call_ret_ty == expected_ty, (
            value_kind, helper_name, spec.ret_ty,
            expected_ty, call_ret_ty)
    for value_kind, (helper_name, new_value_llvm_ty) in (
            llvm_ir._MODIFY_DISPATCH.items()):
        spec = llvm_ir._HELPER_FUNCTIONS[helper_name]
        # MODIFY result is always i32 regardless of value_kind.
        assert spec.ret_ty == "i32", (value_kind, helper_name, spec.ret_ty)
        # The 2nd-arg LLVM type the dispatch table promises must
        # appear in the helper's definition text — a typo on either
        # side surfaces here.
        assert (f"{new_value_llvm_ty} %new_value"
                in spec.definition), (
            value_kind, helper_name, new_value_llvm_ty)


def test_stage206r_polymorphic_dispatch_tables_are_immutable():
    """The two dispatch tables are `MappingProxyType` (read-only
    view) so an op handler cannot accidentally mutate them and
    poison future emissions in the same process. Belt-and-braces
    for the SSOT discipline introduced in audit-fix HIGH-1."""
    with pytest.raises(TypeError):
        llvm_ir._SPLICE_DISPATCH["i32"] = ("evil", "evil", "evil")  # type: ignore[index]
    with pytest.raises(TypeError):
        llvm_ir._MODIFY_DISPATCH["i32"] = ("evil", "evil")  # type: ignore[index]


def test_stage206r_reflect_hash_rejects_zero_operands():
    """v3.1 step 5: REFLECT_HASH is now LOWERED on LLVM (x86 still
    has no arm — catchall stays in place until v3.1 step 6 deletes
    the x86 backend). The lowering needs an i32 handle operand
    naming the cell to hash; zero operands is malformed."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    r = b.emit(tir.OpKind.REFLECT_HASH, result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="REFLECT_HASH takes one operand"):
        llvm_ir.emit_module(mod)


def test_stage206r_reflect_hash_emits_call_to_helper():
    """REFLECT_HASH with an i32 handle operand lowers to a call to
    the `__helix_reflect_hash` helper. The helper pulls in the
    reflection-state global so the cell load resolves at link time."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32())], _i32())
    r = b.emit(tir.OpKind.REFLECT_HASH, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{r.id} = call i32 "
            f"@__helix_reflect_hash(i32 %v{fn.params[0].id})"
            in ll), ll
    # The helper itself is emitted exactly once.
    assert ll.count(
        "define internal i32 @__helix_reflect_hash") == 1, ll
    # Pulled in the reflection-state global.
    assert "@__helix_state_base" in ll, ll


def test_stage206r_reflect_hash_rejects_non_i32_handle():
    """The helper signature is `(i32) -> i32`. An i64 handle would
    produce an LLVM type-mismatch at the call site."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", tir.TIRScalar("i64"))], _i32())
    r = b.emit(tir.OpKind.REFLECT_HASH, fn.params[0], result_ty=_i32())
    b.ret(r)
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="REFLECT_HASH handle has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_reflect_hash_rejects_non_i32_result():
    """The helper returns i32. An i64 result is rejected.

    The REFLECT_HASH op-handler's result-type check fires during
    per-op emission inside `_FnEmitter._emit_op`. It does not
    matter that the function's `ret` returns a different value
    (a CONST_INT, here) — emit_module walks every op in source
    order and fails the first malformed one. Calls to `_emit_op`
    for ops whose result is later unused still run."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    fn = b.begin_function("f", [("h", _i32())], _i32())
    b.emit(tir.OpKind.REFLECT_HASH, fn.params[0],
           result_ty=tir.TIRScalar("i64"))
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(llvm_ir.LLVMEmitError,
                       match="REFLECT_HASH result has LLVM type i64"):
        llvm_ir.emit_module(mod)


def test_stage206r_reflect_hash_helper_has_correct_shape():
    """The `__helix_reflect_hash` helper is a 3-block (entry/load/
    exit) function over the reflection cells: bounds-check, load
    i64 from cell, splitmix64 finalizer, truncate to i32, phi 0
    on OOB. Pin the structural shape so a future drift in the
    finalizer (e.g. a constant typo) surfaces as a test failure
    rather than a runtime collision-rate regression.

    Audit-fix CRITICAL-1: pin the multiplier constants AGAINST their
    hex source-of-truth (not against pre-computed decimals — a
    prior version had a typo that the test happily pinned in
    place). The module-load assertion `_check_reflect_hash_constants`
    is additional defense at the encoding layer."""
    spec = llvm_ir._HELPER_FUNCTIONS["__helix_reflect_hash"]
    assert spec.ret_ty == "i32", spec
    assert spec.module_globals is llvm_ir._HELIX_STATE_GLOBALS, spec
    body = spec.definition
    # Bounds check uses _HELIX_NUM_CELLS.
    assert (f"icmp sge i32 %handle, {llvm_ir._HELIX_NUM_CELLS}"
            in body), body
    # The Stafford mix13 (splitmix64 finalizer) multipliers, derived
    # from their hex bit-pattern source-of-truth.
    assert llvm_ir._SPLITMIX64_C1_HEX == 0xff51afd7ed558ccd
    assert llvm_ir._SPLITMIX64_C2_HEX == 0xc4ceb9fe1a85ec53
    c1_i64 = 0xff51afd7ed558ccd - (1 << 64)
    c2_i64 = 0xc4ceb9fe1a85ec53 - (1 << 64)
    assert f"mul i64 %x1, {c1_i64}" in body, body
    assert f"mul i64 %x2, {c2_i64}" in body, body
    # Three `lshr i64 %.., 33` mixings.
    assert body.count("lshr i64") == 3, body
    # Final truncation to i32.
    assert "trunc i64 %x3 to i32" in body, body
    # Three labels: entry / load / exit. Crude but stable.
    for label in ("entry:", "load:", "exit:"):
        assert label in body, (label, body)


def test_stage206r_reflect_hash_constants_module_load_pinned():
    """Audit-fix CRITICAL-1: the splitmix64 multiplier constants are
    encoded as `hex - (1<<64)` (signed-twos-complement i64) and the
    module-load assertion `_check_reflect_hash_constants` verifies
    the round-trip. A future typo on either constant would crash at
    import rather than silently producing a wrong hash."""
    # Source-of-truth hex matches the well-known Stafford mix13.
    assert llvm_ir._SPLITMIX64_C1_HEX == 0xff51afd7ed558ccd
    assert llvm_ir._SPLITMIX64_C2_HEX == 0xc4ceb9fe1a85ec53
    # The signed encoding round-trips to the hex.
    assert (llvm_ir._SPLITMIX64_C1_I64 + (1 << 64)
            == llvm_ir._SPLITMIX64_C1_HEX)
    assert (llvm_ir._SPLITMIX64_C2_I64 + (1 << 64)
            == llvm_ir._SPLITMIX64_C2_HEX)


def test_stage206r_reflect_hash_round_trip_with_modify():
    """A MODIFY(cell, value) followed by REFLECT_HASH(cell) calls
    both helpers — the SSA chain is `modify ... reflect_hash`. The
    test pins the emission shape so a future refactor that
    accidentally folds the two helpers together is detectable."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    # Verifier function (i32, i32) -> i32 that always accepts.
    b.begin_function("v", [("h", _i32()), ("x", _i32())], _i32())
    b.ret(b.const_int(1))
    b.end_function()
    fn = b.begin_function("f", [("h", _i32()), ("x", _i32())], _i32())
    m = b.emit(tir.OpKind.MODIFY,
               fn.params[0], fn.params[1],
               result_ty=_i32(), attrs={"verifier_fn": "v"})
    # Ignore m's accepted-flag; hash the cell directly.
    h = b.emit(tir.OpKind.REFLECT_HASH, fn.params[0], result_ty=_i32())
    b.ret(h)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    assert (f"%v{m.id} = call i32 @__helix_modify(i32 "
            f"%v{fn.params[0].id}, i32 %v{fn.params[1].id}, "
            f"ptr @v)" in ll), ll
    assert (f"%v{h.id} = call i32 @__helix_reflect_hash(i32 "
            f"%v{fn.params[0].id})" in ll), ll
    # Both helpers emitted.
    assert ll.count("define internal i32 @__helix_modify(") == 1, ll
    assert ll.count(
        "define internal i32 @__helix_reflect_hash(") == 1, ll


def test_stage206r_quote_splice_round_trip_arithmetic():
    """The slot arithmetic in QUOTE / SPLICE matches: QUOTE
    materialises handle = ast_handle % NUM_CELLS; SPLICE indexes
    `state[handle]` directly (no offset). A QUOTE result fed to
    SPLICE addresses the same cell that MODIFY would write to."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("f", [], _i32())
    h = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
               attrs={"ast_handle": 7})
    r = b.emit(tir.OpKind.SPLICE, h, result_ty=_i32())
    b.ret(r)
    b.end_function()
    ll = llvm_ir.emit_module(mod)
    # QUOTE emits the constant; SPLICE calls the helper.
    assert f"%v{h.id} = add i32 0, 7" in ll, ll
    assert (f"%v{r.id} = call i32 @__helix_splice(i32 %v{h.id})"
            in ll), ll


def test_stage206r_quote_splice_modify_is_deterministic():
    def build():
        mod = tir.Module()
        b = tir.IRBuilder(mod)
        b.begin_function("v", [("h", _i32()), ("x", _i32())], _i32())
        b.ret(b.const_int(1))
        b.end_function()
        fn = b.begin_function("f", [("v", _i32())], _i32())
        h = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
                   attrs={"ast_handle": 0})
        b.emit(tir.OpKind.SPLICE, h, result_ty=_i32())
        m = b.emit(tir.OpKind.MODIFY, h, fn.params[0],
                   result_ty=_i32(), attrs={"verifier_fn": "v"})
        b.ret(m)
        b.end_function()
        return mod
    assert llvm_ir.emit_module(build()) == llvm_ir.emit_module(build())


def test_stage206r_state_globals_shared_constant():
    """`_HELIX_STATE_GLOBALS` is referenced by both SPLICE and
    MODIFY helpers — single source of truth."""
    assert llvm_ir._HELIX_STATE_GLOBALS == ("__helix_state_base",)
    for name in ("__helix_splice", "__helix_modify"):
        spec = llvm_ir._HELPER_FUNCTIONS[name]
        assert spec.module_globals is llvm_ir._HELIX_STATE_GLOBALS, (
            name, spec.module_globals)


def test_stage206r_trace_globals_shared_constant():
    """`_HELIX_TRACE_GLOBALS` is referenced by the trace_event
    helper — single source of truth for the (count, buf) pair so a
    future trace-related helper picks up both by reference."""
    assert llvm_ir._HELIX_TRACE_GLOBALS == (
        "__helix_trace_count", "__helix_trace_buf")
    spec = llvm_ir._HELPER_FUNCTIONS["__helix_trace_event"]
    assert spec.module_globals is llvm_ir._HELIX_TRACE_GLOBALS


def test_stage206r_arena_globals_shared_constant():
    """`_HELIX_ARENA_GLOBALS` is referenced by every arena helper —
    typing the name six times invites typo drift; the constant
    eliminates the surface."""
    assert llvm_ir._HELIX_ARENA_GLOBALS == ("__helix_arena_base",)
    # Every arena helper points to the same tuple by reference.
    arena_helpers = (
        "__helix_arena_push", "__helix_arena_push_pair",
        "__helix_arena_push_triple", "__helix_arena_get",
        "__helix_arena_set", "__helix_arena_len",
    )
    for name in arena_helpers:
        spec = llvm_ir._HELPER_FUNCTIONS[name]
        assert spec.module_globals is llvm_ir._HELIX_ARENA_GLOBALS, (
            name, spec.module_globals)


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


# ==========================================================================
# Stage 221 cutover: `--emit-llvm-ir` is the canonical v3.0+ backend
# output. These tests exercise the new CLI flag end-to-end via the
# check.py entry point (parse → typecheck → lower → emit LLVM IR text).
# ==========================================================================
def test_stage221_emit_llvm_ir_smoke(tmp_path):
    """`helixc check --emit-llvm-ir` on a trivial program prints
    LLVM IR text with the target triple and a `define` for main."""
    from helixc.check import main as check_main
    src_path = tmp_path / "smoke.hx"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = check_main(["--emit-llvm-ir", str(src_path)])
    out = buf.getvalue()
    assert rc == 0, out
    assert 'target triple = "x86_64-unknown-linux-gnu"' in out, out
    assert "define i32 @main()" in out, out
    assert "ret i32 42" in out, out


def test_stage221_emit_llvm_ir_in_stdout_modes_mutex(tmp_path):
    """`--emit-llvm-ir` is in the stdout-modes mutex set so combining
    it with `--emit-asm` produces a clean diagnostic (rather than a
    silent first-wins dispatch)."""
    from helixc.check import main as check_main
    src_path = tmp_path / "conflict.hx"
    src_path.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    import io
    import contextlib
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), \
            contextlib.redirect_stderr(err):
        rc = check_main(["--emit-llvm-ir", "--emit-asm",
                         str(src_path)])
    out = err.getvalue() + buf.getvalue()
    assert rc != 0, out
    # The diagnostic names both conflicting flags.
    assert "--emit-llvm-ir" in out, out
    assert "--emit-asm" in out, out


def test_stage221_emit_llvm_ir_in_known_long_flags():
    """The new flag is registered in `_KNOWN_LONG_FLAGS` so an
    unknown-flag check passes for it."""
    from helixc import check as check_module
    assert "--emit-llvm-ir" in check_module._KNOWN_LONG_FLAGS
