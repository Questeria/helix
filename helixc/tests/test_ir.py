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


# Stage 28.9 cycle 108 audit-S regression tests for C107-F1..F8.
# The cycle-107 silent-failures audit flagged 7 _is_i64_type sibling
# sites in x86_64.py that the cycle-106 sweep left on the unsafe
# predicate, plus a catch-all silent-fallthrough in _lower_expr that
# the cycle-106 fix did not generalise.


def test_c107_call_return_u64_stores_full_8_bytes():
    """C107-F2 regression (HIGH conf 90): a CALL whose callee returns
    u64 must store the full 8-byte rax to the caller's result slot
    via `mov [rbp+disp], rax` (48 89 ...). Pre-fix `_is_i64_type` did
    not match u64, so only eax (low 32 bits) was stored — silently
    dropping the high half of the SysV-ABI-delivered return value.
    Also covers C107-F3 (callee RETURN of u64): the callee must load
    via `mov rax, [rbp+disp]` (48 8B 45 ...) — the byte sequence
    `48 8B 45` is the discriminative opcode for the wide load."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn id_u64(x: u64) -> u64 { x }
    fn main() -> i32 {
        let y: u64 = id_u64(1_u64);
        0
    }
    """
    elf = compile_module_to_elf(lower_src(src))
    # `mov rax, [rbp+disp8]` = 48 8B 45 <disp> — required by callee
    # RETURN path (F3) and caller's slot loads.
    assert b"\x48\x8b\x45" in elf, (
        "expected `mov rax, [rbp+disp8]` (48 8B 45 ..) for u64 "
        "load — C107-F3 regression: RETURN still 32-bit"
    )


def test_c107_call_arg_u64_uses_64bit_reg():
    """C107-F1 regression (HIGH conf 90): a CALL passing a u64 arg
    must use the 64-bit `mov rdi, [rbp+disp]` (48 8B 7D <disp>) form.
    Pre-fix only i64/isize args went through INT_REGS_64; u64/usize
    args fell through to `mov edi, [rbp+disp]` (8B 7D <disp>), losing
    the high half of the slot."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn take_u64(x: u64) -> i32 { 0 }
    fn main() -> i32 {
        let v: u64 = 42_u64;
        take_u64(v)
    }
    """
    elf = compile_module_to_elf(lower_src(src))
    # `mov rdi, [rbp+disp8]` = 48 8B 7D <disp> — the 64-bit form for
    # the first SysV int-arg register.
    assert b"\x48\x8b\x7d" in elf, (
        "expected `mov rdi, [rbp+disp8]` (48 8B 7D ..) for u64 CALL "
        "arg — C107-F1 regression: arg still loaded via 32-bit mov edi"
    )


def test_c107_if_else_u64_emits_64bit_branch_copy():
    """C107-F5 regression (HIGH conf 88): an `if c { a_u64 } else
    { b_u64 }` lowers to a merge block with a u64 param. The BR
    block-param copy must use the 64-bit path: `mov rax, [rbp+src]`
    (48 8B 45 ...) + `mov [rbp+dst], rax` (48 89 45 ...). Pre-fix the
    BR arm ran through eax, silently truncating both branches'
    computed u64 values to 32 bits on entry to the merge block."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn pick(c: i32, a: u64, b: u64) -> u64 {
        if c == 0 { a } else { b }
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    # The BR-side merge copy is `mov rax, [rbp+src]` (48 8B 45 disp)
    # then `mov [rbp+dst], rax` (48 89 45 disp). Both 48-prefixed.
    assert b"\x48\x89\x45" in elf, (
        "expected `mov [rbp+disp], rax` (48 89 45 ..) for BR "
        "block-param u64 store — C107-F5 regression: BR still 32-bit"
    )


def test_c107_mut_u64_local_uses_64bit_load_store():
    """C107-F6 regression (HIGH conf 90): `let mut x: u64 = ...; x =
    x + 1u64; x` must use the 64-bit LOAD_VAR/STORE_VAR path. Pre-
    fix every read-modify-write cycle of a mutable u64 local silently
    dropped the high half of the var slot via 32-bit eax-based moves."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn rmw_u64() -> u64 {
        let mut x: u64 = 1_u64;
        x = x + 1_u64;
        x
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    # LOAD_VAR / STORE_VAR for u64 must emit `mov rax, [...]` (48 8B
    # 45) and `mov [...], rax` (48 89 45). Both opcodes are required.
    assert b"\x48\x8b\x45" in elf and b"\x48\x89\x45" in elf, (
        "expected 64-bit LOAD_VAR/STORE_VAR opcodes (48 8B 45 and "
        "48 89 45) for u64 mutable local — C107-F6 regression"
    )


def test_c107_cast_u32_to_u64_uses_zero_extend_not_sign_extend():
    """C107-F7 regression (HIGH conf 85): `x as u64` where x: u32 must
    ZERO-extend, not sign-extend. Pre-fix the predicates `_is_i64_type`
    missed u64 so the cast fell to the bottom 4-byte mov-copy, leaving
    the high 4 bytes of the result slot stale (not zero, not sign-
    extended — uninitialised). Fix adds an unsigned-widening arm that
    emits `mov eax, [src]` + `mov [dst], rax` (relies on x86-64's
    implicit zero-extension of 32-bit destination writes to rax) —
    must NOT emit `movsxd` (48 63 C0)."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn widen_u32(x: u32) -> u64 {
        x as u64
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    # The unsigned-widening arm emits `mov eax, [rbp+src]` (8B 45 ...)
    # followed by `mov [rbp+dst], rax` (48 89 45 ...). The sign-
    # extension form `movsxd rax, eax` = 48 63 C0 must NOT appear in
    # the body of widen_u32.
    assert b"\x48\x63\xc0" not in elf, (
        "movsxd (48 63 C0) found — C107-F7 regression: u32->u64 "
        "took sign-extension path; should be zero-extension via "
        "`mov eax, [..]` + `mov [..], rax`"
    )
    assert b"\x48\x89\x45" in elf, (
        "expected `mov [rbp+disp], rax` (48 89 45 ..) from the "
        "unsigned-widening arm — C107-F7 regression"
    )


def test_c107_char_lit_in_expr_pos_raises_loud():
    """C107-F8 regression (HIGH conf 82): A.CharLit in expression
    position must raise NotImplementedError. Pre-fix the bottom-of-
    _lower_expr `return None` silently dropped CharLit; the caller's
    `or const_int(0)` substitution then folded `'A'` to `0`, making
    downstream `c == 'A'` silently evaluate to `0 == 0 -> true` for
    the wrong reason."""
    src = """
    fn f() -> i32 {
        let c = 'A';
        0
    }
    fn main() -> i32 { 0 }
    """
    try:
        lower_src(src)
    except NotImplementedError as e:
        assert "char" in str(e).lower(), (
            f"NotImplementedError must mention 'char', got: {e}"
        )
        return
    raise AssertionError(
        "A.CharLit must raise NotImplementedError until IR lowering "
        "has a real arm; pre-fix this silently substituted 0"
    )


def test_c107_struct_lit_in_expr_pos_raises_loud():
    """C107-F8 regression (HIGH conf 82): A.StructLit appearing in a
    non-let-stmt expression position (call-arg, if-arm, return value,
    assign rhs) must raise NotImplementedError. The let-stmt path at
    lower_ast.py:848 handles `let x = S{a:1};`; any other expression
    position routed through _lower_expr's bottom `return None`,
    silently folding the StructLit to 0."""
    src = """
    struct Pt { x: i32, y: i32 }
    fn take(p: Pt) -> i32 { p.x }
    fn main() -> i32 {
        take(Pt { x: 1, y: 2 })
    }
    """
    try:
        lower_src(src)
    except NotImplementedError as e:
        assert "struct" in str(e).lower(), (
            f"NotImplementedError must mention 'struct', got: {e}"
        )
        return
    raise AssertionError(
        "A.StructLit in expression position must raise "
        "NotImplementedError; pre-fix this silently substituted 0"
    )


# Stage 28.9 cycle 110 audit-{S, T, CR} regression tests for the
# cycle-109 FINDINGS sweep (1 CRITICAL + 7 HIGH). Each test is
# discriminative: pre-fix it fails, post-fix it passes. See
# docs/audit-stage28-9-cycle109-*.md for the audit citations.


def test_c109_mut_u64_load_store_byte_identical_to_i64():
    """C109 code-review F1 strengthening of C107-F6 (HIGH conf 88):
    `let mut x: u64 = ...; x = ...; x` must produce byte-identical code
    to the i64 version of the same function. The original F6 test
    `test_c107_mut_u64_local_uses_64bit_load_store` is vacuous because
    `ADD u64` and `CONST_INT u64` also emit 48 8B 45 / 48 89 45 — the
    test passes even with F6 reverted (LOAD_VAR/STORE_VAR back to
    32-bit). This sibling pins the discriminator: i64 and u64 fn
    bodies that differ only in the type annotation MUST emit identical
    bytes. Pre-cycle-108 they didn't (i64 used 64-bit path; u64 fell
    to 32-bit). Post-cycle-108 they should match exactly."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src_u64 = """
    fn body_u64(input: u64) -> u64 {
        let mut x: u64 = input;
        x = input;
        x
    }
    fn main() -> i32 { 0 }
    """
    src_i64 = """
    fn body_u64(input: i64) -> i64 {
        let mut x: i64 = input;
        x = input;
        x
    }
    fn main() -> i32 { 0 }
    """
    elf_u64 = compile_module_to_elf(lower_src(src_u64))
    elf_i64 = compile_module_to_elf(lower_src(src_i64))
    assert elf_u64 == elf_i64, (
        f"u64 and i64 fn bodies must produce byte-identical code post "
        f"cycle-108 sweep; sizes u64={len(elf_u64)} i64={len(elf_i64)} "
        f"— C107-F6 strengthening (C109 code-review F1)"
    )


def test_c109_call_return_u64_caller_stores_full_rax():
    """C109 code-review F2 strengthening of C107-F2 (HIGH conf 85):
    a CALL to a u64-returning fn must store the full rax to the
    caller's result slot via `mov [rbp+disp], rax` (48 89 ...). The
    original F2 test only asserted callee RETURN's `mov rax, [rbp+
    disp]` (48 8B 45) which is F3 coverage. F2 is the caller-side
    store after CALL — discriminator is byte-identity with i64 caller
    (both must emit the same 48 89 prefix for the caller-side
    store)."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src_u64 = """
    fn id_u64(x: u64) -> u64 { x }
    fn caller_u64() -> i32 {
        let y: u64 = id_u64(1_u64);
        0
    }
    fn main() -> i32 { 0 }
    """
    src_i64 = """
    fn id_u64(x: i64) -> i64 { x }
    fn caller_u64() -> i32 {
        let y: i64 = id_u64(1_i64);
        0
    }
    fn main() -> i32 { 0 }
    """
    elf_u64 = compile_module_to_elf(lower_src(src_u64))
    elf_i64 = compile_module_to_elf(lower_src(src_i64))
    assert elf_u64 == elf_i64, (
        f"u64 and i64 CALL-return caller storage must be byte-identical "
        f"post cycle-108 sweep; sizes u64={len(elf_u64)} i64={len(elf_i64)} "
        f"— C107-F2 strengthening (C109 code-review F2)"
    )


def test_c110_range_in_value_position_raises_loud():
    """C109-SF-F2 regression (HIGH conf 92): A.Range in a value
    position (e.g. `let r = 0..10;`) must raise NotImplementedError.
    Pre-fix the A.Range arm at lower_ast.py:2006 was an explicit
    silent `return None`; the caller's `or self.builder.const_int(0)`
    then substituted 0 for the lost range, silently miscompiling
    `let r = 0..10; r` to a const-zero. The For iter_expr path
    (line 1820) special-cases Range before reaching _lower_expr so
    Range-in-for-loop still works."""
    src = """
    fn f() -> i32 {
        let r = 0..10;
        0
    }
    fn main() -> i32 { 0 }
    """
    try:
        lower_src(src)
    except NotImplementedError as e:
        assert "range" in str(e).lower(), (
            f"NotImplementedError must mention 'range', got: {e}"
        )
        return
    raise AssertionError(
        "A.Range in value position must raise NotImplementedError; "
        "pre-fix this silently substituted 0"
    )


def test_c110_cast_u32_to_f64_uses_zero_extend_path():
    """C109-SF-F3 regression (HIGH conf 88): `x as f64` where x: u32
    must NOT use the signed 32-bit `cvtsi2sd xmm0, eax` (F2 0F 2A C0).
    For u32 values with the high bit set, signed 32-bit conversion
    produces a negative float (-1.0 for 0xFFFFFFFF) instead of the
    correct unsigned value (~4.29e9). Fix: zero-extend u32→u64 via
    `mov eax, [src]` (implicit zero-extend to rax on x86-64), then
    use the 64-bit REX.W signed `cvtsi2sd xmm0, rax` (F2 48 0F 2A C0).
    With the upper 32 bits known zero, signed-64 conversion gives the
    correct unsigned interpretation. Discriminator: presence of the
    REX.W-prefixed 5-byte sequence F2 48 0F 2A C0 inside the u32-cast
    fn body."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn u32_to_f64(x: u32) -> f64 {
        x as f64
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\xF2\x48\x0F\x2A\xC0" in elf, (
        "expected REX.W cvtsi2sd xmm0, rax (F2 48 0F 2A C0) for u32→f64 "
        "cast — C109-SF-F3 regression: u32 source still signed-32-converts"
    )


def test_c110_cast_u32_to_f32_uses_zero_extend_path():
    """C109-SF-F3 sibling for u32→f32: must emit REX.W cvtsi2ss
    (F3 48 0F 2A C0) after a zero-extending load through eax."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn u32_to_f32(x: u32) -> f32 {
        x as f32
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\xF3\x48\x0F\x2A\xC0" in elf, (
        "expected REX.W cvtsi2ss xmm0, rax (F3 48 0F 2A C0) for u32→f32 "
        "cast — C109-SF-F3 regression"
    )


def test_c111_cast_u64_to_f64_uses_unsigned_high_bit_path():
    """Cycle-111 F1: u64->f64 must not route through signed i64 only.
    High-bit-set u64 values need the unsigned split/convert/double sequence."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn u64_to_f64(x: u64) -> f64 {
        x as f64
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert (
        b"\x48\x89\xc1"          # mov rcx, rax
        b"\x48\xd1\xe8"          # shr rax, 1
        b"\x83\xe1\x01"          # and ecx, 1
        b"\x48\x09\xc8"          # or rax, rcx
        b"\xf2\x48\x0f\x2a\xc0"  # cvtsi2sd xmm0, rax
        b"\xf2\x0f\x58\xc0"      # addsd xmm0, xmm0
    ) in elf, (
        "expected unsigned u64->f64 high-bit path; signed-only cvtsi2sd "
        "misconverts values >= 2^63"
    )


def test_c111_cast_u64_to_f32_uses_unsigned_high_bit_path():
    """Cycle-111 F1 sibling: u64->f32 needs the same unsigned sequence."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn u64_to_f32(x: u64) -> f32 {
        x as f32
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert (
        b"\x48\x89\xc1"          # mov rcx, rax
        b"\x48\xd1\xe8"          # shr rax, 1
        b"\x83\xe1\x01"          # and ecx, 1
        b"\x48\x09\xc8"          # or rax, rcx
        b"\xf3\x48\x0f\x2a\xc0"  # cvtsi2ss xmm0, rax
        b"\xf3\x0f\x58\xc0"      # addss xmm0, xmm0
    ) in elf, (
        "expected unsigned u64->f32 high-bit path; old fallback read only "
        "the low 32 bits"
    )


def test_c112_cast_i64_to_f32_uses_64bit_signed_path():
    """Cycle-112: i64->f32 must emit REX.W cvtsi2ss, not low-32 fallback."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn i64_to_f32(x: i64) -> f32 {
        x as f32
    }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\xf3\x48\x0f\x2a\xc0" in elf, (
        "expected REX.W cvtsi2ss xmm0, rax (F3 48 0F 2A C0) for i64->f32"
    )
    assert b"\xf3\x0f\x2a\xc0" not in elf, (
        "i64->f32 must not use 32-bit cvtsi2ss xmm0, eax"
    )


def test_c110_bit_and_u64_emits_64bit_form():
    """C109-SF-F4 regression (HIGH conf 92): BIT_AND on u64 must emit
    `and rax, rcx` (48 21 C8). Pre-fix the predicate `_is_i64_type`
    only matched i64/isize so u64 fell to 32-bit `and eax, ecx`
    (21 C8 no REX prefix), silently truncating the high half of both
    operands. Discriminator: REX.W-prefixed `and rax, rcx` (48 21 C8)."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn and_u64(a: u64, b: u64) -> u64 { a & b }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\x21\xc8" in elf, (
        "expected REX.W-prefixed `and rax, rcx` (48 21 C8) for u64 "
        "BIT_AND — C109-SF-F4 regression"
    )


def test_c110_bit_or_u64_emits_64bit_form():
    """C109-SF-F4 BIT_OR sibling: `or rax, rcx` = 48 09 C8."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn or_u64(a: u64, b: u64) -> u64 { a | b }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\x09\xc8" in elf, (
        "expected REX.W-prefixed `or rax, rcx` (48 09 C8) for u64 "
        "BIT_OR — C109-SF-F4 regression"
    )


def test_c110_bit_xor_u64_emits_64bit_form():
    """C109-SF-F4 BIT_XOR sibling: `xor rax, rcx` = 48 31 C8."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn xor_u64(a: u64, b: u64) -> u64 { a ^ b }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\x31\xc8" in elf, (
        "expected REX.W-prefixed `xor rax, rcx` (48 31 C8) for u64 "
        "BIT_XOR — C109-SF-F4 regression"
    )


def test_c110_shl_u64_emits_64bit_form():
    """C109-SF-F4 SHL sibling: `shl rax, cl` = 48 D3 E0."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn shl_u64(a: u64, b: u64) -> u64 { a << b }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\xd3\xe0" in elf, (
        "expected REX.W-prefixed `shl rax, cl` (48 D3 E0) for u64 "
        "SHL — C109-SF-F4 regression"
    )


def test_c111_shr_u64_emits_logical_64bit_form():
    """Cycle-111 F2: u64 right shift must be 64-bit logical shr."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn shr_u64(a: u64, b: u64) -> u64 { a >> b }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\xd3\xe8" in elf, (
        "expected REX.W-prefixed `shr rax, cl` (48 D3 E8) for u64 SHR"
    )
    assert b"\x48\xd3\xf8" not in elf, (
        "u64 SHR must not use arithmetic `sar rax, cl`"
    )


def test_c111_shr_usize_emits_logical_64bit_form():
    """Cycle-111 F2 alias: usize right shift follows the u64 SHR path."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn shr_usize(a: usize, b: usize) -> usize { a >> b }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\xd3\xe8" in elf, (
        "expected REX.W-prefixed `shr rax, cl` (48 D3 E8) for usize SHR"
    )


def test_c111_shr_u32_emits_logical_32bit_form():
    """Cycle-111 F2 sibling: unsigned 32-bit right shift uses shr, not sar."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn shr_u32(a: u32, b: u32) -> u32 { a >> b }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\xd3\xe8" in elf, (
        "expected `shr eax, cl` (D3 E8) for u32 SHR"
    )
    assert b"\xd3\xf8" not in elf, (
        "u32 SHR must not use arithmetic `sar eax, cl`"
    )


def test_c110_bit_not_u64_emits_64bit_form():
    """C109-SF-F4 BIT_NOT sibling: `not rax` = 48 F7 D0."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn not_u64(a: u64) -> u64 { ~a }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\xf7\xd0" in elf, (
        "expected REX.W-prefixed `not rax` (48 F7 D0) for u64 "
        "BIT_NOT — C109-SF-F4 regression"
    )


def test_c110_neg_u64_emits_64bit_form():
    """C109-SF-F4 NEG sibling: unary `-a` where a: u64 must emit
    REX.W-prefixed `neg rax` (48 F7 D8), not the 32-bit `neg eax` form."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn neg_u64(a: u64) -> u64 { -a }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\xf7\xd8" in elf, (
        "expected REX.W-prefixed `neg rax` (48 F7 D8) for u64 NEG"
    )


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
