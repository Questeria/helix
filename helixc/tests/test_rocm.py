"""Tests for helixc.backend.rocm — Stage 123 (v2.0 Phase C) substrate.

ROCm / HIP text-emit substrate covering CUDA → AMDGPU op-mapping
coverage + a kernel-emit smoke test.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.rocm import (
    DEFAULT_TARGET,
    ROCM_OBJECT_FORMAT,
    DEFAULT_WAVE_SIZE,
    ROCM_OP_LOWERING,
    HipEmitter,
    lowering_status,
)
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.tile_ir import lower_to_tile, TileOpKind


def test_stage123_module_constants():
    """Stage 123 — MI300 baseline target + wave64 wave size + amdgcn
    object format documented as module constants."""
    assert DEFAULT_TARGET == "gfx942"
    assert ROCM_OBJECT_FORMAT == "amdgcn"
    assert DEFAULT_WAVE_SIZE == 64


def test_stage123_lowering_coverage_complete():
    """Stage 123 — every TileOpKind has a documented lowering entry.

    This is the drift detector that fires at module-load. If a new
    TileOpKind is added to tile_ir.py, this test reminds the dev to
    update the ROCm port table OR mark it skipped with rationale.
    """
    for k in TileOpKind:
        assert k in ROCM_OP_LOWERING, (
            f"TileOpKind {k.name} missing from ROCM_OP_LOWERING — "
            f"add a lowering or mark status='skipped'"
        )


def test_stage123_lowering_status_categories():
    """Stage 123 — every entry's status is one of the documented
    values. Catches typos like 'STUB' vs 'stub'."""
    valid = {"supported", "stub", "deferred", "skipped"}
    for kind, entry in ROCM_OP_LOWERING.items():
        assert entry["status"] in valid, (
            f"TileOpKind {kind.name}: status {entry['status']!r} not in {valid}"
        )


def test_stage123_tma_marked_skipped():
    """Stage 123 — TMA (Hopper-only memory transfer) has no AMD analog;
    must be documented as skipped, not silently routed elsewhere."""
    assert lowering_status(TileOpKind.TMA_LOAD) == "skipped"
    assert lowering_status(TileOpKind.TMA_STORE) == "skipped"


def test_stage123_matmul_status_supported():
    """Stage 124 R6 audit-fix — TILE_MATMUL is supported on ROCm:
    Stage 124 wired MFMA emission (`v_mfma_f32_16x16x16_f16`), see
    `test_stage124_tile_matmul_emits_mfma`. The prior test name was
    `_status_stub` but the assertion checked `== "supported"` — a
    docstring-vs-assertion lie the v2.1 TEST 5-gate caught."""
    assert lowering_status(TileOpKind.TILE_MATMUL) == "supported"


def test_stage123_lowering_status_rejects_non_tileopkind():
    """Stage 123 — lowering_status raises TypeError on non-TileOpKind.
    Mirrors the has_adjoint / adjoint_outputs discipline from
    Stage 117-119 audit-fix.

    v2.3 TEST MED audit-fix: each `pytest.raises(TypeError)` now
    carries a `match=` clause asserting the diagnostic message names
    the actual type — without it, a bare `pytest.raises(TypeError)`
    would pass on ANY TypeError (e.g., one from an unrelated
    `dict.__hash__` call). The match anchors the test to the
    intended error path.
    """
    for bad in ("TILE_MATMUL", 42, None, object(), 3.14, ["TILE_MATMUL"]):
        with pytest.raises(
            TypeError, match=r"lowering_status expects TileOpKind"
        ):
            lowering_status(bad)  # type: ignore[arg-type]


def test_stage123_emit_module_header():
    """Stage 123 — module header emit produces the .amdgcn_target
    directive with the correct triple format."""
    emitter = HipEmitter()
    emitter.emit_module_header()
    out = emitter.buf.getvalue()
    assert ".amdgcn_target" in out
    assert "amdgcn-amd-amdhsa--gfx942" in out


def test_stage123_emit_kernel_stub_smoke():
    """Stage 123 — full emit_module path for a minimal @kernel.

    Per Phase-0 Helix tile-IR lowering: bare `@kernel fn k() {}` produces
    a TileFn with the kernel attr. Substrate emit produces the function
    header + s_endpgm terminator.
    """
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = HipEmitter()
    text = emitter.emit_module(tile_mod)
    assert ".amdgcn_target" in text
    assert ".globl empty_kernel" in text
    assert "s_endpgm" in text


def test_stage123_emit_module_requires_kernel():
    """Stage 123 — emitting a module with no @kernel fn raises (parity
    with PtxEmitter; non-kernel modules don't make sense for AMDGPU)."""
    src = "fn host_only() -> i32 { 0 }"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = HipEmitter()
    with pytest.raises(RuntimeError, match="kernel"):
        emitter.emit_module(tile_mod)


def test_stage123_emit_module_skips_extern_kernels():
    """Stage 123 — extern kernels (is_extern attr) are NOT emitted; they
    are import declarations from the host side. Mirrors PtxEmitter."""
    src = """
    @kernel fn real_kernel() {}
    """
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    # Add a synthetic extern kernel to ensure it's skipped.
    fn = tile_mod.functions["real_kernel"]
    # Synthesize a second kernel with is_extern (parity with the original
    # ptx.py behavior).
    fn2 = type(fn)(
        name="extern_kernel",
        params=[],
        return_ty=fn.return_ty,
        blocks=fn.blocks,
        attrs={"kernel": True, "is_extern": True},
    )
    tile_mod.functions["extern_kernel"] = fn2
    emitter = HipEmitter()
    text = emitter.emit_module(tile_mod)
    assert "real_kernel" in text
    assert "extern_kernel" not in text


def test_stage123_lowering_status_returns_str():
    """Stage 123 — lowering_status always returns a str (never None or
    raises for known kinds)."""
    for k in TileOpKind:
        status = lowering_status(k)
        assert isinstance(status, str)
        assert len(status) > 0


# ============================================================================
# Stage 124 (v2.0 Phase C ROCm wmma) — MFMA + memory + barrier op emit
# ============================================================================
def test_stage124_barrier_wait_emits_swaitcnt_sbarrier():
    """Stage 124 — BARRIER_WAIT lowers to s_waitcnt + s_barrier (parity
    with CUDA __syncthreads → bar.sync 0)."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.BARRIER_WAIT),
            TileOp(kind=TileOpKind.RETURN),
        ])],
        attrs={"kernel": True},
    )
    from helixc.ir.tile_ir import TileModule
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    text = HipEmitter().emit_module(tile_mod)
    assert "s_waitcnt vmcnt(0) lgkmcnt(0)" in text
    assert "s_barrier" in text


def test_stage124_tile_matmul_emits_mfma():
    """Stage 124 — TILE_MATMUL emits v_mfma_f32_16x16x16_f16 (MI300
    MFMA tile-matmul instruction)."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="matmul_k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_MATMUL),
            TileOp(kind=TileOpKind.RETURN),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["matmul_k"] = fn
    text = HipEmitter().emit_module(tile_mod)
    assert "v_mfma_f32_16x16x16_f16" in text


def test_stage124_global_load_store_emits():
    """Stage 124 — TILE_LOAD_GLOBAL / TILE_STORE_GLOBAL emit
    global_load_b128 / global_store_b128 (16-byte tile granularity)."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="memk", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_LOAD_GLOBAL),
            TileOp(kind=TileOpKind.TILE_STORE_GLOBAL),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["memk"] = fn
    text = HipEmitter().emit_module(tile_mod)
    assert "global_load_b128" in text
    assert "global_store_b128" in text


def test_stage124_lds_load_store_emits():
    """Stage 124 — TILE_LOAD_SHARED / TILE_STORE_SHARED emit
    ds_read_b128 / ds_write_b128 (LDS is the AMD analog of CUDA SMEM).

    v2.1 R1 audit-fix Finding H1 (code-reviewer): pre-R1 the emitter
    produced `ds_load_b128` / `ds_store_b128` which are NOT valid
    AMDGPU mnemonics — llvm-mc / hipcc would reject them. The actual
    LDS instructions are `ds_read_b{32,64,128}` / `ds_write_b{...}`.
    Tests passed pre-R1 only because the asserts matched the (wrong)
    emitted token. R1 fixes both sides + asserts the wrong tokens
    do NOT appear (regression-pin).
    """
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="ldsk", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_LOAD_SHARED),
            TileOp(kind=TileOpKind.TILE_STORE_SHARED),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["ldsk"] = fn
    text = HipEmitter().emit_module(tile_mod)
    assert "ds_read_b128" in text
    assert "ds_write_b128" in text
    # Regression-pin: the invalid pre-R1 tokens must NOT appear.
    assert "ds_load_b" not in text
    assert "ds_store_b" not in text


def test_stage124_stub_status_emits_helix_stub_error():
    """v2.1 R1 audit-fix — ops with status="stub" emit a `.error`
    HELIX-STUB directive so hipcc aborts loudly.

    Pre-R1 this test was named `test_stage124_unmapped_op_falls_through_to_comment`
    and claimed to exercise the `; tile-IR op KIND (stub)` fallback
    comment. In reality it exercised the `.error` branch — the test
    name lied. R1 renames + asserts the .error directive explicitly.
    """
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="stubk", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_REDUCE),  # stub status
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["stubk"] = fn
    text = HipEmitter().emit_module(tile_mod)
    assert ".error" in text
    assert "HELIX-STUB" in text
    assert "TILE_REDUCE" in text


# ============================================================================
# Stage 124 R1 audit-fix tests (added 2026-05-19 after the explicit
# 3-clean-audit returned FAIL with 3 HIGH findings):
#   H1 (code-reviewer): ds_load_*/ds_store_* are not valid AMDGPU mnemonics
#   H2 (silent-failure-hunter): status="skipped" silently fell through to
#       a benign comment; TMA on ROCm path produced a no-op kernel
#   H3 (silent-failure-hunter): 5 ops marked status="supported" had no
#       _emit_op branch (phantom-supported); same Stage-120 R2/R3 pattern
# ============================================================================
def test_stage124_r1_skipped_status_emits_helix_skipped_error():
    """R1 H2 — status="skipped" ops (TMA_LOAD, TMA_STORE on the ROCm
    path) must emit a `.error "HELIX-SKIPPED: ..."` directive so a
    miscompile-routing bug is loud, not silent."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    for skipped_kind in (TileOpKind.TMA_LOAD, TileOpKind.TMA_STORE):
        fn = TileFn(
            name="tma_k", params=[], return_ty=None,
            blocks=[TileBlock(id=0, ops=[TileOp(kind=skipped_kind)])],
            attrs={"kernel": True},
        )
        tile_mod = TileModule()
        tile_mod.functions["tma_k"] = fn
        text = HipEmitter().emit_module(tile_mod)
        assert ".error" in text, f"{skipped_kind.name}: expected .error directive"
        assert "HELIX-SKIPPED" in text, (
            f"{skipped_kind.name}: expected HELIX-SKIPPED, "
            f"pre-R1 fell through to benign comment"
        )
        assert skipped_kind.name in text
        # Regression-pin: must NOT silently emit the substrate comment.
        assert f"; tile-IR op {skipped_kind.name} (stub)" not in text


def test_stage124_r1_demoted_ops_emit_helix_stub_error():
    """R1 H3 — TILE_ADD / TILE_SUB / TILE_MUL / TILE_INDEX_LOAD_HBM /
    TILE_INDEX_STORE_HBM were claimed status="supported" pre-R1 but
    had no _emit_op branch (phantom-supported). R1 demotes them to
    status="stub" so the `.error` directive fires.

    This test pins the demoted set: if any of these is later promoted
    back to "supported" without adding an _emit_op branch, the
    exhaustiveness guard's AssertionError will fire and this test
    will explode (which is the intended fail-loudly outcome).
    """
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    demoted = (
        TileOpKind.TILE_ADD,
        TileOpKind.TILE_SUB,
        TileOpKind.TILE_MUL,
        TileOpKind.TILE_INDEX_LOAD_HBM,
        TileOpKind.TILE_INDEX_STORE_HBM,
    )
    for kind in demoted:
        assert lowering_status(kind) == "stub", (
            f"{kind.name}: must be 'stub' after R1 demote, got "
            f"{lowering_status(kind)!r}"
        )
        fn = TileFn(
            name="demok", params=[], return_ty=None,
            blocks=[TileBlock(id=0, ops=[TileOp(kind=kind)])],
            attrs={"kernel": True},
        )
        tile_mod = TileModule()
        tile_mod.functions["demok"] = fn
        text = HipEmitter().emit_module(tile_mod)
        assert ".error" in text and "HELIX-STUB" in text and kind.name in text


def test_stage124_r1_supported_ops_emit_real_instruction():
    """R1 H3 — every status="supported" op must emit a line that is
    neither a comment (`;`) nor a `.error` directive. This is the
    "status-vs-emission consistency" invariant: if the table says
    supported, the emit must produce real AMDGPU instructions.

    RETURN is the documented exception (falls through to the kernel-
    level `s_endpgm` terminator emitted by `emit_kernel_stub`, no
    per-op output). THREAD_IDX is also a documented exception (workitem
    ID is implicit in v0, only an annotation comment is emitted).
    """
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    # Documented per-op no-output / annotation-only exceptions.
    EXCEPTIONS = {TileOpKind.RETURN, TileOpKind.THREAD_IDX}
    for kind, entry in ROCM_OP_LOWERING.items():
        if entry["status"] != "supported":
            continue
        if kind in EXCEPTIONS:
            continue
        fn = TileFn(
            name="supk", params=[], return_ty=None,
            blocks=[TileBlock(id=0, ops=[TileOp(kind=kind)])],
            attrs={"kernel": True},
        )
        tile_mod = TileModule()
        tile_mod.functions["supk"] = fn
        text = HipEmitter().emit_module(tile_mod)
        # The emit MUST contain a real AMDGPU mnemonic line for this
        # supported op — not a `.error` directive (would mean stub
        # leaked into the supported set) and not just kernel framing.
        assert ".error" not in text, (
            f"{kind.name}: status='supported' but emitted .error; "
            f"phantom-supported regression"
        )
        # Strip the kernel-framing lines (.amdgcn_target, .text, .globl,
        # .p2align, .type, label, s_endpgm) — what remains must be the
        # op's emit.
        FRAMING_PREFIXES = (
            ".amdgcn_target", ".text", ".globl", ".p2align", ".type",
            "supk:", "    s_endpgm",
        )
        body_lines = [
            line for line in text.splitlines()
            if line.strip()
            and not any(line.startswith(p) for p in FRAMING_PREFIXES)
        ]
        assert body_lines, f"{kind.name}: no emitted body lines"


def test_stage124_r1_exhaustiveness_guard_fires_on_phantom_supported(monkeypatch):
    """R1 M1 — the exhaustiveness guard at the end of `_emit_op` must
    raise AssertionError if a status="supported" op reaches it (i.e.,
    no `if kind is …` branch matched). This is the second-line defense
    against future re-introduction of the phantom-supported bug."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    import helixc.backend.rocm as rocm_mod

    # Patch a stub op to claim "supported" without adding a branch.
    # TILE_RESHAPE is documented as no-op-at-codegen + status="stub";
    # bumping it to "supported" in the table fakes the phantom case.
    patched = dict(rocm_mod.ROCM_OP_LOWERING)
    patched[TileOpKind.TILE_RESHAPE] = {
        "lowering": "(no-op)",
        "status": "supported",
    }
    monkeypatch.setattr(rocm_mod, "ROCM_OP_LOWERING", patched)

    fn = TileFn(
        name="phantk", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_RESHAPE)])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["phantk"] = fn
    with pytest.raises(AssertionError, match="no codegen branch"):
        rocm_mod.HipEmitter().emit_module(tile_mod)
