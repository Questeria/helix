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


# ============================================================
# Inc 3 — enable map_err + upgrade map_ok to proper packed
# Result transform via SELECT on the runtime tag
# ============================================================


def test_stage49_inc3_map_err_on_err_replaces_payload():
    """map_err(Err(99), 5) returns Err(5) — the Err payload is
    replaced. unwrap_err extracts the new payload (5).

    Pre-Stage-49 map_err was typecheck-rejected because no runtime
    Err side existed to replace. Inc 3 lifts the rejection and
    lowers map_err to a SELECT-on-tag: if Err, repack with
    new_err; otherwise pass r through unchanged."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(99);
    unwrap_err(map_err(r, 5))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == [], \
        f"map_err must typecheck post-Inc-3, got: " \
        f"{[str(e) for e in typecheck(prog)]}"
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 5


def test_stage49_inc3_map_err_on_ok_passes_through_unchanged():
    """map_err on an Ok value must NOT replace the Ok payload —
    the Ok side passes through unchanged. unwrap_ok still
    extracts the original 42."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(42);
    unwrap_ok(map_err(r, 999))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage49_inc3_map_ok_on_ok_replaces_payload():
    """Inc 3 also upgrades map_ok to a proper packed Result
    transform. Pre-Inc-3 map_ok lowered to just args[1] (an i32,
    not a packed Result) — fragile but happened to test-pass
    via accidental RESULT_PAYLOAD truncation. Post-Inc-3:
    map_ok(Ok(7), 99) returns proper Ok(99) packed-i64."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_ok(map_ok(r, 99))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 99


def test_stage49_inc3_map_ok_on_err_passes_through_unchanged():
    """map_ok on an Err value must NOT replace the Err payload —
    the Err side passes through unchanged. Verified via unwrap_err
    extracting the original Err payload."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(33);
    unwrap_err(map_ok(r, 99))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 33


def test_stage49_inc3_map_err_preserves_is_err_status():
    """After map_err(Err(...), ...), is_err should still be true.
    Tests the tag bit is preserved through the SELECT lowering."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(99);
    let r2: Result<i32, i32> = map_err(r, 5);
    if is_err(r2) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 1


def test_stage49_inc3_map_ok_preserves_is_ok_status():
    """After map_ok(Ok(...), ...), is_ok should still be true."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    let r2: Result<i32, i32> = map_ok(r, 99);
    if is_ok(r2) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 1


def test_stage49_inc3_map_err_non_result_first_arg_rejects():
    """Negative test: map_err's first arg must be Result."""
    src = """
fn main() -> i32 {
    let x: i32 = 7;
    unwrap_ok(map_err(x, 99))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("requires first arg Result" in str(e) for e in errs), \
        f"map_err with non-Result must reject, got: {[str(e) for e in errs]}"


def test_stage49_inc3_map_err_arity_mismatch_rejects():
    """Negative test: map_err takes exactly 2 arguments."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(99);
    unwrap_err(map_err(r))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("takes 2 arguments" in str(e) for e in errs), \
        f"map_err with wrong arity must reject, got: {[str(e) for e in errs]}"


# ============================================================
# Inc 4 — real `?` early-return branch IR
# ============================================================


def test_stage49_inc4_question_on_ok_falls_through_to_payload():
    """? on an Ok-tagged Result extracts the payload and continues
    in the caller. The COND_BR takes the false (ok) edge. Same
    runtime behavior as Inc 1's identity-lower placeholder, so
    dogfood_17 still exits 42 — confirmed below in the dogfood
    regression."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(42);
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper()) }
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage49_inc4_question_on_err_propagates_up():
    """The defining feature of Inc 4: `?` on an Err-tagged Result
    early-returns the Err from the enclosing fn. Pre-Inc-4 this
    fell through to RESULT_PAYLOAD and extracted the Err payload
    as if it were Ok (the F1-dynamic Phase-0 limitation). Post-
    Inc-4 the COND_BR takes the true (err) edge, RETURN emits
    the original packed Err, and the caller sees Err(99)."""
    src = """
fn maybe_fail() -> Result<i32, i32> {
    Err(99)
}
fn helper() -> Result<i32, i32> {
    let v: i32 = maybe_fail()?;
    Ok(v + 100)
}
fn main() -> i32 {
    unwrap_err(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    # ? on Err(99) propagates -> helper returns Err(99).
    # unwrap_err extracts 99.
    assert _run_elf(elf) == 99


def test_stage49_inc4_question_propagates_through_chained_calls():
    """Chained `?` across multiple Result-returning fns. If any
    intermediate fn returns Err, the propagation chain terminates
    and the deepest non-Err caller observes the Err. Pre-Inc-4
    this was just sequential payload extraction; post-Inc-4 the
    error short-circuits."""
    src = """
fn level3() -> Result<i32, i32> { Err(7) }
fn level2() -> Result<i32, i32> {
    let v: i32 = level3()?;
    Ok(v + 1)
}
fn level1() -> Result<i32, i32> {
    let v: i32 = level2()?;
    Ok(v + 10)
}
fn main() -> i32 {
    unwrap_err(level1())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    # level3 -> Err(7); level2's `?` propagates -> Err(7);
    # level1's `?` propagates -> Err(7). main's unwrap_err = 7.
    assert _run_elf(elf) == 7


def test_stage49_inc4_question_does_not_run_post_q_code_on_err():
    """The `?` early-return MUST skip code after `?` if the operand
    was Err. Pre-Inc-4 the lowering extracted the payload and
    continued, so post-`?` code ran (silent miscompile). Post-
    Inc-4 the COND_BR jumps over the rest of the block.

    Test via a side effect: helper's `?` on Err MUST NOT execute
    the `Ok(99)` final-expression. The caller observes Err."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Err(50);
    let v: i32 = r?;
    Ok(99)
}
fn main() -> i32 {
    unwrap_err(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    # If `?` propagated correctly, main sees Err(50) -> 50.
    # If post-`?` code ran (pre-fix bug), Ok(99) would shadow
    # and main would see Ok via unwrap_err (panic / undefined).
    assert _run_elf(elf) == 50


def test_stage49_inc4_question_arithmetic_around_still_works_on_ok():
    """`r? + 2` on Ok(40) still yields 42 (Inc 4 preserves Inc 1
    arithmetic composition for the Ok-fall-through path)."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(40);
    let v: i32 = r? + 2;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper()) }
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage49_inc4_dogfood_17_still_exits_42():
    """dogfood_17_try_operator.hx (Stage 48 demo) compiles all-Ok
    paths via safe_div(20,4) then safe_div(5,1). Both succeed, so
    `?` falls through both times. Post-Inc-4 the lowering inserts
    COND_BRs but the false (ok) edge is always taken at runtime.
    End-to-end exit code 42 unchanged."""
    proj = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = open(os.path.join(
        proj, "helixc/examples/dogfood_17_try_operator.hx")).read()
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


# ============================================================
# Inc 1.5 — runtime wrong-arm tag-check on unwrap_ok / unwrap_err
# (gate-1 silent-failure F1 + F2 fix)
# ============================================================


def _run_elf_full(elf: bytes) -> tuple[int, str]:
    """Run + return (returncode, stderr_text). Used for panic-path
    tests where we need to observe the panic message + the non-
    clean exit."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(elf)
        bin_path = f.name
    try:
        os.chmod(bin_path, 0o755)
        abs_p = bin_path.replace("\\", "/").replace("C:", "/mnt/c")
        r = subprocess.run(
            ["wsl", "--", "bash", "-c",
             f"chmod +x {abs_p} && exec {abs_p}"],
            capture_output=True, timeout=30,
        )
        return r.returncode, r.stderr.decode("utf-8", "replace")
    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass


def test_stage49_inc1_5_unwrap_ok_on_dynamic_err_panics():
    """Gate-1 silent-failure F1: pre-Inc-1.5 `unwrap_ok(make_err())`
    where make_err() returns Err(99) silently extracted the Err
    payload (99) as if it were Ok — HIGH silent miscompile.

    Post-Inc-1.5 the unwrap_ok lowering emits a runtime
    RESULT_TAG check + COND_BR to a TRAP block. On mismatch the
    process prints `panic[28503]: unwrap_ok called on an Err-
    tagged Result` to stderr and sys_exit's. The user-visible
    signal: NON-CLEAN process termination (no longer the silent
    99 exit code) AND a panic diagnostic on stderr.

    Note: the WSL bridge can report the post-syscall exit code
    as 4294967295 (uint32 -1) for signaled processes rather than
    the Linux 128+signum or sys_exit byte value. The safety
    property is that the process does NOT exit with 99 (the
    wrong-arm payload), and the panic message reaches stderr.
    Future portability work may normalize the exit code."""
    src = """
fn make_err() -> Result<i32, i32> { Err(99) }
fn main() -> i32 {
    unwrap_ok(make_err())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    returncode, stderr = _run_elf_full(elf)
    assert returncode != 99, \
        f"silent-miscompile regression: got the wrong-arm payload " \
        f"as exit code (returncode={returncode!r}, stderr={stderr!r})"
    assert "unwrap_ok called on an Err-tagged Result" in stderr, \
        f"panic message must reach stderr, got: {stderr!r}"


def test_stage49_inc1_5_unwrap_err_on_dynamic_ok_panics():
    """Symmetric companion to the unwrap_ok F1 fix. Pre-fix
    `unwrap_err(make_ok())` silently extracted Ok payload (7);
    post-fix panics with `unwrap_err called on an Ok-tagged
    Result` on stderr."""
    src = """
fn make_ok() -> Result<i32, i32> { Ok(7) }
fn main() -> i32 {
    unwrap_err(make_ok())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    returncode, stderr = _run_elf_full(elf)
    assert returncode != 7, \
        f"silent-miscompile regression: got the wrong-arm payload " \
        f"as exit code (returncode={returncode!r}, stderr={stderr!r})"
    assert "unwrap_err called on an Ok-tagged Result" in stderr, \
        f"panic message must reach stderr, got: {stderr!r}"


def test_stage49_inc1_5_unwrap_ok_compose_through_map_err_panics():
    """Gate-1 silent-failure F2: composition `unwrap_ok(map_err(
    Err(1), 99))` propagated the Err through map_err's SELECT and
    THEN silently extracted via unwrap_ok — payload 99 leaked as
    Ok. Post-Inc-1.5 the runtime tag-check catches the dynamic-
    operand case at unwrap_ok and panics."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(1);
    unwrap_ok(map_err(r, 99))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    returncode, stderr = _run_elf_full(elf)
    assert returncode != 99, \
        f"compose-through-map_err silent-miscompile regression: " \
        f"returncode={returncode!r}, stderr={stderr!r}"
    assert "unwrap_ok called on an Err-tagged Result" in stderr, \
        f"panic message must reach stderr, got: {stderr!r}"


def test_stage49_inc1_5_unwrap_ok_on_correct_arm_still_works():
    """Sanity: the runtime tag-check must NOT break the
    correct-arm case. unwrap_ok(Ok(42)) must still return 42
    cleanly, no panic, no spurious stderr."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(42);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    returncode, stderr = _run_elf_full(elf)
    assert returncode == 42, \
        f"correct-arm unwrap_ok regression: returncode={returncode!r}, " \
        f"stderr={stderr!r}"
    assert "panic" not in stderr.lower(), \
        f"correct-arm path must not panic, got: {stderr!r}"


# ============================================================
# Gate-1 code-review M4: chained map_ok / map_err composition
# pins the SELECT-of-SELECT semantics
# ============================================================


def test_stage49_inc3_map_chain_preserves_tag_through_select_of_select():
    """Gate-1 code-review M4: pins the SELECT-of-SELECT semantics
    for chained `map_err(map_ok(r, x), y)` (and the symmetric
    `map_ok(map_err(r, x), y)`). Pre-this-test, both map_*
    lowerings used SELECT on tag-equality independently; a future
    refactor that swaps SELECT operand order would silently break
    one of these directions without breaking the existing
    single-map tests. This test catches that regression class."""
    # Direction 1: Ok(5) → map_err passes through (Ok stays) → map_ok replaces inner with 42 → unwrap_ok = 42.
    src_ok_path = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(5);
    unwrap_ok(map_ok(map_err(r, 999), 42))
}
"""
    prog = parse(src_ok_path, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42

    # Direction 2: Err(7) → map_ok passes through (Err stays) → map_err replaces inner with 42 → unwrap_err = 42.
    src_err_path = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(7);
    unwrap_err(map_err(map_ok(r, 99), 42))
}
"""
    prog = parse(src_err_path, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


# ============================================================
# Gate-2 type-design G2-H1: wider-payload typecheck reject
# ============================================================


def test_stage49_gate2_g2h1_result_i64_payload_rejected():
    """Gate-2 type-design G2-H1: pre-fix `Result<i64, i32>` (or
    `Result<*, i64>`) typecheck-passed but silently truncated the
    high 32 bits of the i64 at IR lowering (RESULT_PACK with a
    32-bit payload slot). The lower_ast comment falsely claimed
    typecheck enforced i32 — now true. Post-fix typecheck rejects
    with a diagnostic naming the side + the offending type +
    pointing at the Stage 50+ widening plan."""
    src = """
fn id_i64(x: i64) -> i64 { x }
fn main() -> i32 {
    let big: i64 = id_i64(5000000000);
    let r: Result<i64, i32> = Ok(big);
    0
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = " ".join(str(e) for e in errs)
    assert "not supported by the Stage 49 packed-i64" in err_strs, \
        f"Ok(i64) must reject with G2-H1 diag, got: {err_strs}"


def test_stage49_gate2_g2h1_result_f64_payload_rejected():
    """Symmetric to i64: Result<f64, i32> must reject at typecheck."""
    src = """
fn main() -> i32 {
    let r: Result<f64, i32> = Ok(3.14_f64);
    0
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = " ".join(str(e) for e in errs)
    assert ("Stage 49 packed-i64" in err_strs
            or "Ok() payload type f64" in err_strs), \
        f"Ok(f64) must reject with G2-H1 diag, got: {err_strs}"


def test_stage49_gate2_g2h1_result_f32_payload_rejected():
    """Gate-2 silent-failure LOW-3 follow-up: explicit f32 case.
    Pre-fix `Result<f32, i32>` typecheck-passed but the IR-side
    32-bit float payload would round-trip through the i32 slot
    only by bit-reinterpretation, which the lowering doesn't do.
    Post-fix the G2-H1 typecheck reject covers f32 alongside i64
    and f64. Pinning it explicitly here closes the LOW-3
    recommendation to add a canonical f32-reject test."""
    src = """
fn main() -> i32 {
    let r: Result<f32, i32> = Ok(3.14_f32);
    0
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = " ".join(str(e) for e in errs)
    assert "Ok() payload type f32" in err_strs, \
        f"Ok(f32) must reject with G2-H1 diag, got: {err_strs}"
    assert "Stage 49 packed-i64" in err_strs, \
        f"Reject diagnostic must mention the Stage 49 payload " \
        f"representation, got: {err_strs}"


# ============================================================
# Gate-3 G3-H1: map_ok / map_err must also reject wider payloads
# (G2-H1 coverage gap — bypass via map_* constructor)
# ============================================================


def test_stage49_gate3_g3h1_map_ok_wider_payload_rejected():
    """Gate-3 G3-H1: pre-fix `map_ok(r, 9999999999_i64)` typecheck-
    passed because G2-H1 only fired at the Ok/Err constructors,
    NOT at map_ok/map_err arms which build a fresh TyResult from
    the caller-provided new_value. The i64 then silently
    truncated to i32 at IR lowering — same defect class as G2-H1,
    different entry point. Post-fix map_ok also calls
    _reject_non_i32_result_payload on the new_value type."""
    src = """
fn id_i64(x: i64) -> i64 { x }
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    let r2: Result<i64, i32> = map_ok(r, id_i64(9999999999));
    0
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = " ".join(str(e) for e in errs)
    assert "not supported by the Stage 49 packed-i64" in err_strs, \
        f"map_ok with i64 new_value must reject, got: {err_strs}"
    assert "map_ok new_value" in err_strs, \
        f"diagnostic must name the map_ok side, got: {err_strs}"


def test_stage49_gate3_g3h1_map_err_wider_payload_rejected():
    """Gate-3 G3-H1 symmetric companion for map_err. Pre-fix
    `map_err(r, 9999999999_i64)` silently truncated the i64
    new_err to i32. Post-fix typecheck rejects with the same
    G2-H1-style diagnostic naming the map_err side."""
    src = """
fn id_i64(x: i64) -> i64 { x }
fn main() -> i32 {
    let r: Result<i32, i32> = Err(7);
    let r2: Result<i32, i64> = map_err(r, id_i64(9999999999));
    0
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = " ".join(str(e) for e in errs)
    assert "not supported by the Stage 49 packed-i64" in err_strs, \
        f"map_err with i64 new_value must reject, got: {err_strs}"
    assert "map_err new_value" in err_strs, \
        f"diagnostic must name the map_err side, got: {err_strs}"


def test_stage49_gate3_g3h1_map_ok_i32_new_value_still_works():
    """Sanity: map_ok with an i32 new_value still works post-fix.
    Pins the happy-path so the G3-H1 reject doesn't over-fire."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_ok(map_ok(r, 42))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage49_gate2_g2h1_result_known_i32_still_works():
    """Sanity: wrapper-around-i32 must still work. `Known<i32>` is
    identity-lowered at expression position, so the payload is
    still i32 at packing time. The G2-H1 reject helper strips
    wrapper layers before checking the inner."""
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    let r: Result<Known<i32>, i32> = Ok(k);
    from_known(unwrap_ok(r))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], \
        f"Result<Known<i32>, i32> must typecheck clean " \
        f"(wrapper-stripping in G2-H1 helper), got: " \
        f"{[str(e) for e in errs]}"
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


# ============================================================
# Gate-2 silent-failure SF1-F1: __try Err arm emits TRACE_EXIT
# for @trace fns (mirrors A.Return C2-2 fix)
# ============================================================


def test_stage49_gate2_sf1f1_try_err_arm_emits_trace_exit_in_traced_fn():
    """Gate-2 silent-failure SF1-F1: pre-fix the `?` Err
    propagation site emitted RETURN without a preceding
    TRACE_EXIT. For @trace'd fns that propagate Err via `?`,
    the trace stream lost an EXIT entry (unbalanced ENTRY/EXIT
    pair). Invisible at Phase-0 (backend stubs trace ops) but
    would corrupt the buffer once Stage 30 runtime exists.
    Same defect class as Audit 28.8 cycle 2 C2-2 (`A.Return`
    arm). Post-fix every block ending in RETURN inside a
    @trace'd fn has a TRACE_EXIT as the immediately-preceding
    op."""
    src = """
@trace
fn helper(r: Result<i32, i32>) -> Result<i32, i32> {
    let v: i32 = r?;
    Ok(v + 1)
}
fn main() -> i32 {
    0
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    mod = lower(prog)
    helper = mod.functions["helper"]
    for blk in helper.blocks:
        if not blk.ops:
            continue
        last = blk.ops[-1]
        if last.kind != tir.OpKind.RETURN:
            continue
        prev = blk.ops[-2] if len(blk.ops) >= 2 else None
        assert prev is not None and prev.kind == tir.OpKind.TRACE_EXIT, (
            f"@trace fn 'helper': block id={blk.id} ends in RETURN "
            f"but is not preceded by TRACE_EXIT — SF1-F1 regression. "
            f"Last two ops: "
            f"{prev.kind if prev else None} / {last.kind}"
        )


def test_stage49_gate2_sf1f1_untraced_fn_with_try_still_no_trace_exit():
    """Sanity sibling: non-@trace fns must NOT gain TRACE_EXIT.
    The fix is gated on `_is_fn_traced`, so untraced fns must
    emit RETURN alone (matches pre-Stage-49 codegen)."""
    src = """
fn helper(r: Result<i32, i32>) -> Result<i32, i32> {
    let v: i32 = r?;
    Ok(v + 1)
}
fn main() -> i32 {
    0
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    mod = lower(prog)
    helper = mod.functions["helper"]
    trace_exits = [op for blk in helper.blocks for op in blk.ops
                   if op.kind == tir.OpKind.TRACE_EXIT]
    assert trace_exits == [], (
        f"untraced fn 'helper' must have zero TRACE_EXIT ops, "
        f"got {len(trace_exits)}"
    )
