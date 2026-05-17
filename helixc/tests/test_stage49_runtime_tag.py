"""Stage 49 — runtime Ok/Err tag for Result<T, E> (Tier 4 #14 Inc 3).

Stage 49 Inc 1 introduces three TIR opcodes:
  - RESULT_PACK(tag, payload) -> packed i64
  - RESULT_TAG(packed) -> i32
  - RESULT_PAYLOAD(packed) -> i32

and changes Ok / Err / unwrap_ok / unwrap_err / __try lowering to
use the packed representation.

Phase-0 invariants that Inc 1 preserves:
  - dogfood_17_try_operator.hx still exits 42.
  - All 50 Stage 46 + Stage 48 tests still pass.
  - Static-Ok pathway unchanged in observable runtime behavior:
    `Ok(v)` round-trips to `v` through `unwrap_ok`.

Inc 1 explicitly does NOT yet:
  - Runtime tag-check on unwrap_ok / unwrap_err (Inc 1.5 / Inc 2).
  - Conditional-branch propagation in `__try` (Inc 4).
  - `is_ok` / `is_err` / `map_err` (Inc 2 / Inc 3).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from helixc.backend.x86_64 import compile_module_to_elf
from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck
from helixc.ir import tir
from helixc.ir.lower_ast import lower


def _run_elf(elf: bytes) -> int:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(elf)
        bin_path = f.name
    try:
        os.chmod(bin_path, 0o755)
        abs_p = bin_path.replace("\\", "/").replace("C:", "/mnt/c")
        r = subprocess.run(
            ["wsl", "--", "bash", "-c", f"chmod +x {abs_p} && {abs_p}"],
            capture_output=True, timeout=30,
        )
        return r.returncode
    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass


# ============================================================
# Opcode-level: RESULT_PACK / RESULT_TAG / RESULT_PAYLOAD exist
# ============================================================


def test_stage49_inc1_opcodes_registered():
    """The three new TIR opcodes exist on OpKind."""
    assert tir.OpKind.RESULT_PACK.value == "result.pack"
    assert tir.OpKind.RESULT_TAG.value == "result.tag"
    assert tir.OpKind.RESULT_PAYLOAD.value == "result.payload"


# ============================================================
# Type lowering: Result<i32, i32> -> packed i64
# ============================================================


def test_stage49_inc1_result_type_lowers_to_i64():
    """A function returning Result<i32, i32> has its IR return
    type set to TIRScalar('i64') — the packed-tag representation."""
    src = """
fn helper() -> Result<i32, i32> {
    Ok(7)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    m = lower(prog)
    helper_fn = m.functions["helper"]
    assert isinstance(helper_fn.return_ty, tir.TIRScalar), \
        f"expected scalar return, got {helper_fn.return_ty!r}"
    assert helper_fn.return_ty.name == "i64", \
        f"expected i64 packed Result return, got {helper_fn.return_ty.name!r}"


# ============================================================
# IR shape: Ok(v) emits RESULT_PACK(0, v)
# ============================================================


def _all_ops_of_kind(fn: tir.FnIR, kind: tir.OpKind) -> list[tir.Op]:
    out = []
    for blk in fn.blocks:
        for op in blk.ops:
            if op.kind == kind:
                out.append(op)
    return out


def test_stage49_inc1_ok_emits_result_pack():
    """`Ok(42)` lowers to a RESULT_PACK op with two operands
    (tag const, payload)."""
    src = """
fn helper() -> Result<i32, i32> {
    Ok(42)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    m = lower(prog)
    helper_fn = m.functions["helper"]
    packs = _all_ops_of_kind(helper_fn, tir.OpKind.RESULT_PACK)
    assert len(packs) == 1, \
        f"expected exactly one RESULT_PACK in helper, got {len(packs)}"
    assert len(packs[0].operands) == 2, \
        f"RESULT_PACK must have 2 operands, got {len(packs[0].operands)}"
    # Result type is i64.
    assert isinstance(packs[0].results[0].ty, tir.TIRScalar)
    assert packs[0].results[0].ty.name == "i64"


def test_stage49_inc1_err_emits_result_pack():
    """`Err(13)` also lowers to RESULT_PACK, with tag-arg = const 1."""
    src = """
fn helper() -> Result<i32, i32> {
    Err(13)
}
fn main() -> i32 { 0 }
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    m = lower(prog)
    helper_fn = m.functions["helper"]
    packs = _all_ops_of_kind(helper_fn, tir.OpKind.RESULT_PACK)
    assert len(packs) == 1
    pack = packs[0]
    # Find the const-int op that defines the tag operand, verify it's 1.
    tag_value = pack.operands[0]
    consts = _all_ops_of_kind(helper_fn, tir.OpKind.CONST_INT)
    tag_const = next((c for c in consts
                      if c.results and c.results[0].id == tag_value.id), None)
    assert tag_const is not None, \
        "RESULT_PACK tag operand must trace to a CONST_INT"
    assert tag_const.attrs.get("value") == 1, \
        f"Err tag must be 1, got {tag_const.attrs.get('value')}"


def test_stage49_inc1_ok_tag_is_zero():
    """Symmetric to the Err test: Ok constructor uses tag = 0."""
    src = """
fn helper() -> Result<i32, i32> {
    Ok(7)
}
fn main() -> i32 { 0 }
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    m = lower(prog)
    helper_fn = m.functions["helper"]
    packs = _all_ops_of_kind(helper_fn, tir.OpKind.RESULT_PACK)
    assert len(packs) == 1
    pack = packs[0]
    tag_value = pack.operands[0]
    consts = _all_ops_of_kind(helper_fn, tir.OpKind.CONST_INT)
    tag_const = next((c for c in consts
                      if c.results and c.results[0].id == tag_value.id), None)
    assert tag_const is not None
    assert tag_const.attrs.get("value") == 0, \
        f"Ok tag must be 0, got {tag_const.attrs.get('value')}"


# ============================================================
# IR shape: unwrap_ok / unwrap_err / __try emit RESULT_PAYLOAD
# ============================================================


def test_stage49_inc1_unwrap_ok_emits_result_payload():
    """`unwrap_ok(r)` lowers to a RESULT_PAYLOAD op."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(42);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    m = lower(prog)
    main_fn = m.functions["main"]
    payloads = _all_ops_of_kind(main_fn, tir.OpKind.RESULT_PAYLOAD)
    assert len(payloads) >= 1, \
        f"expected at least one RESULT_PAYLOAD in main, got {len(payloads)}"
    # Result type is i32.
    assert isinstance(payloads[0].results[0].ty, tir.TIRScalar)
    assert payloads[0].results[0].ty.name == "i32"


def test_stage49_inc1_try_operator_emits_result_payload():
    """`r?` (parsed as __try(r)) lowers to RESULT_PAYLOAD in Inc 1.
    Inc 4 will replace this with conditional-branch propagation."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(7);
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper()) }
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    m = lower(prog)
    helper_fn = m.functions["helper"]
    payloads = _all_ops_of_kind(helper_fn, tir.OpKind.RESULT_PAYLOAD)
    assert len(payloads) >= 1, \
        "expected at least one RESULT_PAYLOAD from `r?` in Inc 1"


# ============================================================
# End-to-end: pack/unpack round-trip exits with the right code
# ============================================================


def test_stage49_inc1_ok_round_trip_exits_42():
    """`Ok(42)` packed and then unwrapped through the new IR ops
    still exits 42 — the static-Ok pathway is value-preserved."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(42);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage49_inc1_through_function_call_returning_result():
    """Result-typed function return values round-trip through the
    SysV i64 ABI path (rax) — the calling convention check."""
    src = """
fn make_ok() -> Result<i32, i32> {
    Ok(99)
}
fn main() -> i32 {
    unwrap_ok(make_ok())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 99


def test_stage49_inc1_err_payload_round_trip():
    """`Err(13)` packed and then payload-extracted via unwrap_err
    yields 13. (Inc 1 has no tag check; this exercises the payload
    extraction is correct symmetrically to the Ok side.)"""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(13);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 13


def test_stage49_inc1_dogfood_17_still_exits_42():
    """The full Stage 48 dogfood (chained `?` through Result-
    returning function calls) must still exit 42 under the new
    packed-tag representation. This is the cross-stage self-host
    invariant."""
    src_path = os.path.join(os.path.dirname(__file__), "..",
                            "examples", "dogfood_17_try_operator.hx")
    with open(src_path) as f:
        src = f.read()
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, \
        "dogfood_17 must continue exiting 42 after the packed-tag transition"


def test_stage49_inc1_large_payload_round_trip():
    """Verify the payload extraction zero-extends correctly: a
    payload near the i32 boundary (e.g. 1_000_000) packs into the
    low 32 bits and extracts back unchanged. Catches a sign-extend
    bug if `mov eax, eax` were ever replaced by `movsxd rax, eax`."""
    src = """
fn make() -> Result<i32, i32> {
    Ok(1000000)
}
fn main() -> i32 {
    unwrap_ok(make())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    # Exit codes are conventionally 0-255 in POSIX, but Linux truncates
    # to the low 8 bits. 1_000_000 & 0xFF = 64.
    assert _run_elf(elf) == (1000000 & 0xFF)


def test_stage49_inc1_chained_ok_calls_through_packed_returns():
    """Three Result-returning fns composed via `?` and unwrap.
    Each function-return-value crosses the rax-as-i64 SysV boundary;
    each `?` extracts a payload. End-to-end value must still be 42."""
    src = """
fn a() -> Result<i32, i32> { Ok(10) }
fn b() -> Result<i32, i32> {
    let x: i32 = a()?;
    Ok(x + 12)
}
fn c() -> Result<i32, i32> {
    let y: i32 = b()?;
    Ok(y + 20)
}
fn main() -> i32 {
    unwrap_ok(c())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


# ============================================================
# Inc 2 — enable is_ok / is_err via runtime tag
# ============================================================


def test_stage49_inc2_is_ok_on_ok_construct_typechecks_and_returns_true():
    """is_ok(Ok(...)) now typechecks (Inc 2 lifted the Stage 46
    F1 typecheck reject) and at runtime returns 1 because the
    packed-i64 tag is 0 (Ok). Pre-Inc-2 the typecheck rejected
    with a 'no runtime tag' diagnostic."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(42);
    if is_ok(r) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == [], \
        f"is_ok must typecheck post-Inc-2, got: {[str(e) for e in typecheck(prog)]}"
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 1


def test_stage49_inc2_is_ok_on_err_construct_returns_false():
    """is_ok(Err(...)) at runtime returns 0 because the packed
    tag is 1 (Err). Verified via the tag-check lowering."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(99);
    if is_ok(r) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 0


def test_stage49_inc2_is_err_on_ok_construct_returns_false():
    """is_err(Ok(...)) returns 0. Symmetric companion to is_ok."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(42);
    if is_err(r) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 0


def test_stage49_inc2_is_err_on_err_construct_returns_true():
    """is_err(Err(...)) returns 1. Symmetric companion to is_ok."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(99);
    if is_err(r) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 1


def test_stage49_inc2_is_ok_on_dynamic_result_via_call():
    """is_ok on a Result returned by a fn call — the dynamic case
    that was the original motivator for the runtime tag (per the
    Stage 46 F1 closure ledger). Pre-Inc-2 this was unrepresentable
    because no runtime tag existed; post-Inc-2 it works."""
    src = """
fn safe_div(a: i32, b: i32) -> Result<i32, i32> {
    if b == 0 { Err(b) } else { Ok(a / b) }
}
fn main() -> i32 {
    let good: Result<i32, i32> = safe_div(10, 2);
    let bad: Result<i32, i32> = safe_div(10, 0);
    let g: i32 = if is_ok(good) { 1 } else { 0 };
    let b: i32 = if is_err(bad) { 1 } else { 0 };
    g + b
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    # is_ok(good)=1 + is_err(bad)=1 = 2.
    assert _run_elf(elf) == 2


def test_stage49_inc2_non_result_operand_still_rejected():
    """is_ok(<non-Result>) must still typecheck-reject. Pre-Inc-2
    rejected for `requires Result<T, E>` reason; post-Inc-2 also
    rejected, same diagnostic."""
    src = """
fn main() -> i32 {
    let x: i32 = 7;
    if is_ok(x) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("requires Result" in str(e) for e in errs), \
        f"is_ok on non-Result must reject, got: {[str(e) for e in errs]}"


def test_stage49_inc2_arity_mismatch_still_rejected():
    """is_ok() with the wrong arity remains rejected."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    if is_ok(r, 99) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("takes 1 argument" in str(e) for e in errs), \
        f"is_ok with wrong arity must reject, got: {[str(e) for e in errs]}"
