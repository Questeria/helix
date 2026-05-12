"""Tests for helixc.ir.lower_ast (Tensor IR lowering)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir import tir


def lower_src(src: str) -> tir.Module:
    return lower(parse(src))


def test_empty_function():
    mod = lower_src("fn nothing() {}")
    assert "nothing" in mod.functions
    fn = mod.functions["nothing"]
    assert len(fn.params) == 0
    assert isinstance(fn.return_ty, tir.TIRUnit)
    # Should end with a return op
    assert any(op.kind == tir.OpKind.RETURN for op in fn.entry.ops)


def test_arith_function():
    mod = lower_src("fn add(a: i32, b: i32) -> i32 { a + b }")
    fn = mod.functions["add"]
    assert len(fn.params) == 2
    assert isinstance(fn.return_ty, tir.TIRScalar)
    assert fn.return_ty.name == "i32"
    # Should have an ADD op
    assert any(op.kind == tir.OpKind.ADD for op in fn.entry.ops)


def test_constant_int():
    mod = lower_src("fn k() -> i32 { 42 }")
    fn = mod.functions["k"]
    consts = [op for op in fn.entry.ops if op.kind == tir.OpKind.CONST_INT]
    assert len(consts) == 1
    assert consts[0].attrs["value"] == 42


def test_constant_float():
    mod = lower_src("fn k() -> f32 { 3.14 }")
    fn = mod.functions["k"]
    consts = [op for op in fn.entry.ops if op.kind == tir.OpKind.CONST_FLOAT]
    assert len(consts) == 1
    assert abs(consts[0].attrs["value"] - 3.14) < 1e-6


def test_let_binding():
    mod = lower_src("fn f() -> i32 { let x = 7; x + x }")
    fn = mod.functions["f"]
    # Let binds x to a const(7); then x + x should reuse v_x twice in the ADD
    add_ops = [op for op in fn.entry.ops if op.kind == tir.OpKind.ADD]
    assert len(add_ops) == 1
    assert add_ops[0].operands[0] == add_ops[0].operands[1]


def test_call():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn main() -> i32 { double(5) }
    """
    mod = lower_src(src)
    main = mod.functions["main"]
    calls = [op for op in main.entry.ops if op.kind == tir.OpKind.CALL]
    assert len(calls) == 1
    assert calls[0].attrs["target"] == "double"


def test_nested_calls():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(double(3), 4) }
    """
    mod = lower_src(src)
    main = mod.functions["main"]
    calls = [op for op in main.entry.ops if op.kind == tir.OpKind.CALL]
    assert len(calls) == 2
    targets = [c.attrs["target"] for c in calls]
    assert "double" in targets
    assert "add" in targets


def test_tensor_type_lowered():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]> {
        a
    }
    """
    mod = lower_src(src)
    fn = mod.functions["matmul"]
    # Param 0 should be a TensorTy with f32 dtype and 2 dim vars
    p0_ty = fn.params[0].ty
    assert isinstance(p0_ty, tir.TIRTensorTy)
    assert p0_ty.dtype.name == "f32"
    assert len(p0_ty.shape) == 2
    assert isinstance(p0_ty.shape[0], tir.DimVar) and p0_ty.shape[0].name == "N"


def test_tile_type_lowered():
    src = "fn k(x: tile<bf16, [16, 16], smem>) {}"
    mod = lower_src(src)
    fn = mod.functions["k"]
    p0_ty = fn.params[0].ty
    assert isinstance(p0_ty, tir.TIRTileTy)
    assert p0_ty.dtype.name == "bf16"
    assert p0_ty.memspace == "smem"


def test_kernel_attribute():
    src = "@kernel fn k() {}"
    mod = lower_src(src)
    fn = mod.functions["k"]
    assert fn.attrs.get("kernel") is True


def test_if_lowered_to_cfg():
    src = "fn f(b: bool) -> i32 { if b { 1 } else { 2 } }"
    mod = lower_src(src)
    fn = mod.functions["f"]
    # CFG-based lowering creates extra blocks (then/else/merge) and
    # emits cond_br + br ops
    cond_brs = [op for blk in fn.blocks for op in blk.ops
                if op.kind == tir.OpKind.COND_BR]
    brs = [op for blk in fn.blocks for op in blk.ops
           if op.kind == tir.OpKind.BR]
    assert len(cond_brs) == 1
    assert len(brs) >= 2  # one from each arm to merge
    # Merge block should have a single param for the if-result
    assert len(fn.blocks) >= 4  # entry + then + else + merge


def test_unary_neg():
    src = "fn f() -> i32 { -42 }"
    mod = lower_src(src)
    fn = mod.functions["f"]
    negs = [op for op in fn.entry.ops if op.kind == tir.OpKind.NEG]
    assert len(negs) == 1


def test_unique_value_ids():
    src = "fn f() -> i32 { 1 + 2 + 3 }"
    mod = lower_src(src)
    fn = mod.functions["f"]
    all_ids = []
    for op in fn.entry.ops:
        for r in op.results:
            all_ids.append(r.id)
    assert len(all_ids) == len(set(all_ids)), "all SSA value ids must be unique"


def test_c76_f1_for_range_i64_increment_dtype_matches_iterator():
    """Stage 28.9 cycle 77 audit-T F1 regression (HIGH conf 78):
    when the for-range iterator is i64, the `+= 1` increment must
    construct the constant `1` with dtype i64, NOT default i32. Pre-fix
    the increment emitted `CONST_INT(1, ty=i32)` then `ADD(i64-cur,
    i32-one, result_ty=i64)`, which the x86_64 backend dispatched
    by result type — it issued an 8-byte read of the i32 slot,
    leaking 4 bytes of uninitialized stack into every loop step."""
    src = """
    fn loop_i64() -> i32 {
        let mut total: i64 = 0_i64;
        for _i in 0_i64 .. 5_i64 {
            total += 7_i64;
        }
        total as i32
    }
    """
    mod = lower_src(src)
    fn = mod.functions["loop_i64"]
    # Find the CONST_INT(1) inside the for-range body's increment block.
    # There may be multiple CONST_INT ops; the increment-step one is the
    # one whose result_ty matches the iterator (i64) and value=1.
    increment_ones = []
    for blk in fn.blocks:
        for op in blk.ops:
            if (op.kind == tir.OpKind.CONST_INT
                    and op.attrs.get("value") == 1
                    and isinstance(op.results[0].ty, tir.TIRScalar)
                    and op.results[0].ty.name == "i64"):
                increment_ones.append(op)
    assert increment_ones, (
        "expected at least one CONST_INT(value=1, ty=i64) for the "
        "for-range increment in an i64-typed iterator; pre-fix dtype "
        "defaulted to i32 and mismatched the ADD result_ty"
    )


def test_c96_loop_blocks_appended_to_fn_blocks():
    """Stage 28.9 cycle 97 audit-T C96-1 regression (HIGH conf 90):
    `loop { body }` lowering must `append_block` (not `new_block`)
    its header + body blocks. Pre-fix `new_block` created Block
    instances detached from `current_fn.blocks`, so the orphaned
    blocks were invisible to slot pre-allocation, label emission,
    and BR target lookup — backend aborted at "BR to unknown
    block <id>" for any loop expression."""
    src = """
    fn main() -> i32 {
        let mut x = 0;
        loop { x = x + 1; }
    }
    """
    mod = lower_src(src)
    fn = mod.functions["main"]
    # Pre-fix: only the entry block exists in fn.blocks (loop's blocks
    # were orphaned). Post-fix: entry + header + body = 3 blocks min.
    assert len(fn.blocks) >= 3, (
        f"expected >=3 blocks (entry + loop header + body); got "
        f"{len(fn.blocks)} — A.Loop blocks may be orphaned"
    )
    # Verify any BR target in the function points at a block that
    # actually exists in fn.blocks.
    block_ids = {b.id for b in fn.blocks}
    for blk in fn.blocks:
        for op in blk.ops:
            if op.kind == tir.OpKind.BR:
                target_id = op.attrs.get("target_block")
                if target_id is not None:
                    assert target_id in block_ids, (
                        f"BR target {target_id} not in fn.blocks "
                        f"{sorted(block_ids)}"
                    )


def test_c100_unsigned_cmp_emits_setb_not_setl():
    """Stage 28.9 cycle 100 regression (HIGH conf 92): unsigned int
    compares must emit `setb` (0F 92) not `setl` (0F 9C). Cycle-99
    F2 caught the bug: signed setcc on high-bit-set u32 values
    miscompiles (`0xFFFFFFFF_u32 < 1_u32` returns true under signed
    cmp). Cycle-100 added `unsigned_int_cmp_setters` with setb/setbe/
    seta/setae. This test inspects the emitted ELF for the unsigned
    opcodes when the operand type is u32."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn cmp_u32(a: u32, b: u32) -> i32 {
        if a < b { 1 } else { 0 }
    }
    fn main() -> i32 { cmp_u32(0_u32, 1_u32) }
    """
    elf = compile_module_to_elf(lower_src(src))
    # setb al = 0F 92 C0; setl al = 0F 9C C0. Either appears in
    # int cmp fn bodies — assert the unsigned setb is present.
    assert b"\x0f\x92\xc0" in elf, (
        "expected unsigned setb opcode (0F 92 C0) for u32 cmp — "
        "cycle-100 regression: signed setl still emitted"
    )


def test_c102_u64_add_emits_64bit_path():
    """Stage 28.9 cycle 102 regression (HIGH conf 92): u64/usize
    arithmetic must take the 64-bit codegen path (rex.W + rax/rcx).
    Cycle-101 F2 caught the bug: ADD/SUB/MUL only checked
    `_is_i64_type`, so u64/usize silently fell through to the
    32-bit path and truncated. Cycle-102 introduced
    `_is_64bit_int_type` and routed all three to it.

    This test asserts the emitted ELF for a `u64 + u64` body
    contains the rex.W ADD opcode (48 01 C8 = `add rax, rcx`)."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn add_u64(a: u64, b: u64) -> u64 {
        a + b
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    # `add rax, rcx` = 48 01 C8. The 64-bit ADD path emits this;
    # the 32-bit path emits `add eax, ecx` = 01 C8 (no rex.W).
    assert b"\x48\x01\xc8" in elf, (
        "expected rex.W ADD opcode (48 01 C8) for u64 add — "
        "cycle-102 regression: u64 still falls through to 32-bit path"
    )


def test_c105_f64_to_f32_cast_emits_cvtsd2ss():
    """Stage 28.9 cycle 106 audit-T C105-F1 regression (HIGH conf 90):
    `f64 as f32` must emit `cvtsd2ss` (F2 0F 5A C0) — pre-fix this
    fell through to a 4-byte mov-copy, silently emitting the wrong
    bit-pattern for the narrowing cross-precision cast."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn narrow(x: f64) -> f32 {
        x as f32
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\xf2\x0f\x5a\xc0" in elf, (
        "expected cvtsd2ss opcode (F2 0F 5A C0) for f64 -> f32 cast — "
        "C105-F1 regression: 4-byte mov-copy still silently emitted"
    )


def test_c105_f32_to_f64_cast_emits_cvtss2sd():
    """C105-F1 regression: symmetric widening cast must emit
    `cvtss2sd` (F3 0F 5A C0)."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn widen(x: f32) -> f64 {
        x as f64
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\xf3\x0f\x5a\xc0" in elf, (
        "expected cvtss2sd opcode (F3 0F 5A C0) for f32 -> f64 cast"
    )


def test_c105_u64_const_emits_64bit_path():
    """Stage 28.9 cycle 106 audit-R C105-F1 regression (HIGH conf 92):
    a u64 CONST_INT must emit 8-byte mov_rax_imm64 + mov_mem_rbp_rax.
    Pre-fix `_is_i64_type` matched only i64/isize, so u64 CONST_INT
    emitted 32-bit `mov eax, imm32` into an 8-byte slot — leaving
    high 4 bytes stale."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn make_u64() -> u64 {
        12345_u64
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    # `mov rax, imm64` = 48 B8 ... — rex.W + opcode B8 + 8 imm bytes.
    # Look for the prefix `48 B8` which is the discriminative marker.
    assert b"\x48\xb8" in elf, (
        "expected `mov rax, imm64` opcode (48 B8) for u64 CONST_INT — "
        "C105-F1 regression: u64 still falls through to 32-bit imm"
    )


def test_c105_break_in_loop_raises_loud():
    """Stage 28.9 cycle-105 silent-failure F1 regression (CRITICAL conf 95):
    pre-fix `loop { ...; if c { break; } }` typechecked + lowered to an
    infinite loop because A.Break fell through the _lower_expr catch-all
    `return None`. Fix raises NotImplementedError at the lowering site so
    a silent miscompile becomes a loud build failure until real break/
    continue CFG support lands."""
    src = """
    fn f() -> i32 {
        let mut sum: i32 = 0;
        loop { sum = sum + 1; if sum >= 5 { break; } }
        sum
    }
    """
    try:
        lower_src(src)
    except NotImplementedError as e:
        assert "break" in str(e).lower(), (
            f"NotImplementedError must mention 'break', got: {e}"
        )
        return
    raise AssertionError(
        "A.Break must raise NotImplementedError until CFG support exists; "
        "pre-fix this silently emitted an infinite loop"
    )


def test_c105_continue_in_loop_raises_loud():
    """Stage 28.9 cycle-105 silent-failure F1 regression (CRITICAL conf 95):
    companion to A.Break test — A.Continue must also raise loudly."""
    src = """
    fn f() -> i32 {
        let mut sum: i32 = 0;
        let mut i: i32 = 0;
        while i < 10 { i = i + 1; if i == 3 { continue; } sum = sum + 1; }
        sum
    }
    """
    try:
        lower_src(src)
    except NotImplementedError as e:
        assert "continue" in str(e).lower(), (
            f"NotImplementedError must mention 'continue', got: {e}"
        )
        return
    raise AssertionError(
        "A.Continue must raise NotImplementedError until CFG support exists"
    )


def test_c105_unit_return_type_compatible():
    """Stage 28.9 cycle-105 type-design F105-1 regression (HIGH conf 90):
    pre-fix `fn foo() -> () {}` raised a spurious 'type error: ()
    does not match ()' because source-typed `()` resolved to
    TyPrim('()') while implicit-unit body produced TyUnit(), and the
    cross-class pair failed `_compatible`. Fix normalizes TyName('()')
    to TyUnit() in _resolve_type."""
    from helixc.frontend.typecheck import typecheck
    from helixc.frontend.parser import parse as _parse
    src = "fn foo() -> () { }\nfn main() -> i32 { 0 }\n"
    typecheck(_parse(src))


def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
