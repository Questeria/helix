"""Tests for helixc.backend.proof_manifest — Stage 122 (v2.0 Phase C.3)
attestation-binding ProofObligation manifest emitter.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import pytest

from helixc.backend.proof_manifest import (
    PROOF_MANIFEST_VERSION,
    extract_enclave_tag,
    fn_proof_obligations,
    emit_manifest,
    serialize_manifest,
    verify_manifest_hash,
)
from helixc.frontend.typecheck import (
    typecheck, TypeChecker,
    TyEnclave, TyConf, TyPrim,
)
from helixc.frontend.parser import parse


def test_stage122_extract_enclave_tag_innermost():
    """extract_enclave_tag walks layered wrappers to find the
    innermost TyEnclave tag."""
    # Bare TyEnclave.
    t1 = TyEnclave(enclave="sgx", inner=TyPrim("i32"))
    assert extract_enclave_tag(t1) == "sgx"

    # TyConf wrapping TyEnclave (Conf as outer; should still find enclave).
    t2 = TyConf(level="high", inner=TyEnclave(enclave="tdx", inner=TyPrim("f32")))
    assert extract_enclave_tag(t2) == "tdx"

    # No enclave anywhere.
    t3 = TyConf(level="med", inner=TyPrim("i32"))
    assert extract_enclave_tag(t3) is None

    # Bare primitive.
    assert extract_enclave_tag(TyPrim("i32")) is None


def test_stage122_extract_enclave_tag_bounded_recursion():
    """extract_enclave_tag terminates even on pathological deep nesting
    (defensive: cycle/very-deep guards at seen<32)."""
    # Construct 64-deep nesting (exceeds the 32-iter cap).
    t = TyPrim("i32")
    for _ in range(64):
        t = TyConf(level="med", inner=t)
    # Should not loop forever — returns None safely.
    assert extract_enclave_tag(t) is None


def test_stage122_fn_obligations_basic():
    """fn_proof_obligations extracts purity/effects/enclave from
    a typechecked program."""
    src = """
    @pure
    fn add(x: i32, y: i32) -> i32 { x + y }

    @effect(io)
    fn log(x: i32) -> i32 { x }

    fn make_secret() -> InEnclaveSGX<i32> { __wrap_enclave(42) }

    fn main() -> i32 { 0 }
    """
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    # Pure adder.
    o_add = fn_proof_obligations(tc.functions["add"])
    assert o_add["is_pure"] is True
    assert o_add["effects"] == []
    assert o_add["enclave_tag"] is None
    assert o_add["param_count"] == 2

    # IO-effecting log.
    o_log = fn_proof_obligations(tc.functions["log"])
    assert o_log["is_pure"] is False
    assert "io" in o_log["effects"]

    # Enclave-returning fn.
    o_secret = fn_proof_obligations(tc.functions["make_secret"])
    assert o_secret["enclave_tag"] == "sgx"


def test_stage122_emit_manifest_structure():
    """emit_manifest returns the documented schema and computes a
    manifest_sha256 over the canonical form."""
    src = """
    @pure
    fn p() -> i32 { 0 }

    fn main() -> i32 { 0 }
    """
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m = emit_manifest(tc.functions, helix_version="v2.0-test")

    # Schema check.
    assert m["format_version"] == PROOF_MANIFEST_VERSION
    assert m["helix_version"] == "v2.0-test"
    assert m["function_count"] == 2
    assert isinstance(m["functions"], list)
    assert m["signature"] is None  # substrate; signing deferred
    assert "manifest_sha256" in m
    assert len(m["manifest_sha256"]) == 64  # sha256 hex digest

    # Functions are sorted by name.
    names = [f["name"] for f in m["functions"]]
    assert names == sorted(names)


def test_stage122_manifest_hash_is_deterministic():
    """Two emit_manifest calls on the same module produce the same
    manifest_sha256 (sorted-key canonicalization). Critical for
    attestation reproducibility."""
    src = """
    fn a() -> i32 { 1 }
    fn b() -> i32 { 2 }
    fn main() -> i32 { 0 }
    """
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m1 = emit_manifest(tc.functions, helix_version="v2.0-test")
    m2 = emit_manifest(tc.functions, helix_version="v2.0-test")
    assert m1["manifest_sha256"] == m2["manifest_sha256"]


def test_stage122_manifest_hash_changes_with_artifact():
    """Different artifact_sha256 produces a different manifest_sha256.
    A regulator who pins the manifest hash thus pins both the type-
    system proof obligations AND the compiled binary."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m1 = emit_manifest(tc.functions, artifact_sha256="aa" * 32)
    m2 = emit_manifest(tc.functions, artifact_sha256="bb" * 32)
    assert m1["manifest_sha256"] != m2["manifest_sha256"]


def test_stage122_verify_manifest_hash_self_consistency():
    """A freshly-emitted manifest passes verify_manifest_hash."""
    src = """
    fn main() -> i32 { 0 }
    @pure fn h() -> i32 { 0 }
    """
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m = emit_manifest(tc.functions)
    assert verify_manifest_hash(m) is True


def test_stage122_verify_manifest_hash_detects_tampering():
    """Tampering with manifest body invalidates verify_manifest_hash."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m = emit_manifest(tc.functions)
    assert verify_manifest_hash(m)

    # Mutate a field (an attacker editing the proof obligations).
    m["functions"][0]["is_pure"] = not m["functions"][0]["is_pure"]
    assert verify_manifest_hash(m) is False


def test_stage122_verify_manifest_hash_missing_field():
    """A manifest without manifest_sha256 fails verification (not
    silently True)."""
    m = {"format_version": PROOF_MANIFEST_VERSION, "functions": []}
    assert verify_manifest_hash(m) is False


def test_stage122_serialize_manifest_canonical():
    """serialize_manifest produces byte-identical output for the
    same manifest dict (sorted keys + fixed separators)."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m = emit_manifest(tc.functions)
    s1 = serialize_manifest(m)
    s2 = serialize_manifest(m)
    assert s1 == s2

    # Round-trip via json.
    parsed = json.loads(s1)
    assert parsed["manifest_sha256"] == m["manifest_sha256"]


def test_stage122_serialize_manifest_compact_form():
    """indent=None produces the compact bytes-canonical form a
    verifier would re-canonicalize during attestation challenge."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m = emit_manifest(tc.functions)
    compact = serialize_manifest(m, indent=None)
    pretty = serialize_manifest(m, indent=2)
    assert len(compact) < len(pretty)
    # Both should parse to the same dict.
    assert json.loads(compact) == json.loads(pretty)


def test_stage122_enclave_param_flagged():
    """has_enclave_param is True if any parameter has an enclave tag,
    used by downstream audit tooling that wants to track enclave-typed
    params separately from enclave-typed returns."""
    src = """
    fn consume_secret(s: InEnclaveTDX<i32>) -> i32 { __exit_enclave(s) }
    fn main() -> i32 { 0 }
    """
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    o = fn_proof_obligations(tc.functions["consume_secret"])
    assert o["has_enclave_param"] is True
    # Return is i32 (bare), so enclave_tag is None.
    assert o["enclave_tag"] is None
