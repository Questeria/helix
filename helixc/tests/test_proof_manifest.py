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
    FunctionObligation,
    ProofManifest,
    Sha256Hex,
    SignatureFormat,
    as_sha256_hex,
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


def test_stage122_extract_enclave_tag_bounded_recursion_raises():
    """v2.2 polish item 12 (BE LOW-1 from v2.1 5-clean-gate) — prior
    behavior was to silently return None on type-chains exceeding the
    32-depth cap. That collapsed two distinct outcomes:
      (a) chain terminates with no enclave tag → None
      (b) chain exceeds depth-32 → None
    R1 fix distinguishes them: case (b) now raises ValueError with
    full diagnostic so a producer with a cyclic / pathologically deep
    type chain surfaces as a bug instead of a silent miscategorization."""
    # Construct 64-deep nesting (exceeds the 32-iter cap).
    t = TyPrim("i32")
    for _ in range(64):
        t = TyConf(level="med", inner=t)
    with pytest.raises(ValueError, match="depth exceeded"):
        extract_enclave_tag(t)


def test_stage122_extract_enclave_tag_at_depth_cap():
    """v2.2 polish item 12 — exactly-at-cap (depth 32) succeeds; the
    raise happens at depth 33 (strict overflow, not boundary-strict)."""
    # Construct 32-deep nesting — bare-primitive at innermost.
    # Loop walks 32 times finding TyConf.inner=...; the 33rd iteration
    # finds a TyPrim and returns None cleanly.
    t = TyPrim("i32")
    for _ in range(32):
        t = TyConf(level="med", inner=t)
    # At depth exactly 32: the loop terminates naturally on the TyPrim
    # at the bottom (no `inner` attr) without hitting the depth cap.
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
    assert o_add.is_pure is True
    assert o_add.effects == ()
    assert o_add.enclave_tag is None
    assert o_add.param_count == 2

    # IO-effecting log.
    o_log = fn_proof_obligations(tc.functions["log"])
    assert o_log.is_pure is False
    assert "io" in o_log.effects

    # Enclave-returning fn.
    o_secret = fn_proof_obligations(tc.functions["make_secret"])
    assert o_secret.enclave_tag == "sgx"


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

    # Schema check (v2.4 item 3 slice 2/3 — frozen ProofManifest).
    assert isinstance(m, ProofManifest)
    assert m.format_version == PROOF_MANIFEST_VERSION
    assert m.helix_version == "v2.0-test"
    assert m.function_count == 2
    assert isinstance(m.functions, tuple)
    assert m.signature is None  # substrate; signing deferred
    assert len(m.manifest_sha256) == 64  # sha256 hex digest

    # Functions are sorted by name.
    names = [f.name for f in m.functions]
    assert names == sorted(names)


def test_v23_signature_format_enum_default_is_deferred():
    """v2.3 item 3 slice 3 — emit_manifest defaults signature_format
    to SignatureFormat.DEFERRED; the manifest field stores the Enum's
    `.value` (plain str) so the JSON wire form is unchanged from the
    pre-Enum substrate."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    m = emit_manifest(tc.functions)
    assert m.signature_format == "ed25519-or-rsa-pss-sha256-DEFERRED"
    assert m.signature_format == SignatureFormat.DEFERRED.value
    # The stored value must be a plain str (JSON-serializable, hash-
    # stable) — not an Enum member repr.
    assert type(m.signature_format) is str


def test_v23_signature_format_enum_accepts_member_and_str():
    """v2.3 item 3 slice 3 — emit_manifest accepts a SignatureFormat
    member OR a valid str; both coerce to the same stored value."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    # Enum member.
    m1 = emit_manifest(tc.functions,
                       signature_format=SignatureFormat.ED25519)
    assert m1.signature_format == "ed25519"
    # Plain str (coerced through the Enum).
    m2 = emit_manifest(tc.functions, signature_format="ed25519")
    assert m2.signature_format == "ed25519"
    assert m1.signature_format == m2.signature_format


def test_v23_signature_format_enum_rejects_unknown_scheme():
    """v2.3 item 3 slice 3 — an unrecognized signing-scheme string
    raises ValueError at emit_manifest's boundary, not deep inside a
    downstream attestation verifier."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    with pytest.raises(ValueError, match="not a recognized signing"):
        emit_manifest(tc.functions, signature_format="ecdsa-p384")
    with pytest.raises(ValueError, match="not a recognized signing"):
        emit_manifest(tc.functions, signature_format="ED25519")  # case


def test_v23_signature_format_enum_members():
    """v2.3 item 3 slice 3 — closed-set membership: the Enum has
    exactly the 4 documented schemes."""
    assert {s.name for s in SignatureFormat} == {
        "DEFERRED", "ED25519", "RSA_PSS_SHA256", "TEE_QUOTE",
    }
    # str-based Enum: members compare equal to their string value.
    assert SignatureFormat.ED25519 == "ed25519"
    assert SignatureFormat.TEE_QUOTE == "tee-quote"


def test_v23_signature_format_manifest_still_verifies():
    """v2.3 item 3 slice 3 — a manifest emitted with a non-default
    signature_format still passes verify_manifest_hash (the field is
    inside the canonical hash; the Enum `.value` keeps it hash-stable)."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    m = emit_manifest(tc.functions,
                      signature_format=SignatureFormat.TEE_QUOTE)
    assert verify_manifest_hash(m) is True


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
    assert m1.manifest_sha256 == m2.manifest_sha256


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
    assert m1.manifest_sha256 != m2.manifest_sha256


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
    """Tampering with manifest body invalidates verify_manifest_hash.

    v2.4 item 3 slice 2/3: the producer-side ProofManifest is frozen,
    so the tamper is applied to the wire dict (`to_dict()`) — which is
    exactly what an attestation verifier holds: a dict deserialized
    from untrusted JSON, not the producer's frozen dataclass."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    m = emit_manifest(tc.functions)
    assert verify_manifest_hash(m)

    # An attacker editing the proof obligations in the wire dict.
    d = m.to_dict()
    d["functions"][0]["is_pure"] = not d["functions"][0]["is_pure"]
    assert verify_manifest_hash(d) is False


def test_stage122_verify_manifest_hash_missing_field_raises():
    """v2.2 polish item 3 (BE MED-3 from v2.1 5-clean-gate) — a
    manifest without the `manifest_sha256` field raises ValueError
    (not silently returns False). Prior behavior collapsed
    'malformed manifest' and 'tampered manifest' into the same
    False return. Attestation verifiers downstream could not
    distinguish them. R1 fix: structural absence raises,
    False is reserved for actual hash mismatch (tamper signal)."""
    m = {"format_version": PROOF_MANIFEST_VERSION, "functions": []}
    with pytest.raises(ValueError, match="missing required"):
        verify_manifest_hash(m)


def test_stage122_verify_manifest_hash_empty_field_raises():
    """v2.2 polish item 3 — same disambiguation: an empty / falsy
    `manifest_sha256` field also raises (structural defect, not
    tamper)."""
    m = {
        "format_version": PROOF_MANIFEST_VERSION,
        "functions": [],
        "manifest_sha256": "",
    }
    with pytest.raises(ValueError, match="empty or falsy"):
        verify_manifest_hash(m)


def test_stage122_verify_manifest_hash_non_dict_raises():
    """v2.2 5-clean-gate BE LOW-1 audit-fix — non-dict input (e.g.
    JSON parser produced wrong shape) previously raised generic
    TypeError from membership test. R1 fix surfaces ValueError
    with malformed-manifest diagnostic, parity with other paths."""
    with pytest.raises(ValueError, match="expected dict"):
        verify_manifest_hash(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected dict"):
        verify_manifest_hash([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected dict"):
        verify_manifest_hash("not a dict")  # type: ignore[arg-type]


def test_stage122_verify_manifest_hash_wrong_type_field_raises():
    """v2.2 5-clean-gate BE MEDIUM-1 audit-fix — non-string
    `manifest_sha256` (int, list, dict) previously fell through to
    `claimed == computed` returning False — collapsing "wrong-type
    manifest_sha256" with "actual hash mismatch (tamper)". R1
    fix surfaces ValueError before the comparison."""
    for bad in (12345, ["x", "y"], {"k": "v"}, 3.14):
        m = {
            "format_version": PROOF_MANIFEST_VERSION,
            "functions": [],
            "manifest_sha256": bad,
        }
        with pytest.raises(ValueError, match="must be a str"):
            verify_manifest_hash(m)


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
    assert parsed["manifest_sha256"] == m.manifest_sha256


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


def test_v23_as_sha256_hex_accepts_canonical_digest():
    """v2.3 polish item 3 (slice 1) — `as_sha256_hex` accepts a
    canonical 64-char lowercase hex digest (the output shape of
    `hashlib.sha256(...).hexdigest()`) and returns the value typed
    as `Sha256Hex` (runtime-equivalent to str)."""
    import hashlib
    digest = hashlib.sha256(b"helix").hexdigest()
    assert len(digest) == 64
    tagged = as_sha256_hex(digest)
    # NewType is runtime-equivalent to str.
    assert tagged == digest
    assert isinstance(tagged, str)


def test_v23_as_sha256_hex_rejects_non_string():
    """v2.3 polish item 3 — non-string input raises ValueError with
    a clear diagnostic. Mirrors the verify_manifest_hash type-guard
    parity established in the v2.2 5-clean-gate."""
    for bad in (None, 12345, b"\x00" * 32, ["aa"] * 32, {"hex": "aa"}):
        with pytest.raises(ValueError, match="expected str"):
            as_sha256_hex(bad)  # type: ignore[arg-type]


def test_v23_as_sha256_hex_rejects_wrong_length():
    """v2.3 polish item 3 — strings shorter or longer than 64 chars
    are rejected. Catches the common producer mistake of passing
    a md5 (32) or sha1 (40) digest into a sha256 slot."""
    for bad in ("aa", "aa" * 16, "aa" * 20, "aa" * 33, ""):
        with pytest.raises(ValueError, match="not a 64-char lowercase hex"):
            as_sha256_hex(bad)


def test_v23_as_sha256_hex_rejects_non_hex_chars():
    """v2.3 polish item 3 — exactly-64-char strings with non-hex
    characters are rejected."""
    bad = "g" * 64  # 'g' is outside [0-9a-f]
    with pytest.raises(ValueError, match="not a 64-char lowercase hex"):
        as_sha256_hex(bad)
    bad2 = "z" + "a" * 63
    with pytest.raises(ValueError, match="not a 64-char lowercase hex"):
        as_sha256_hex(bad2)


def test_v23_as_sha256_hex_rejects_uppercase():
    """v2.3 polish item 3 — uppercase / mixed-case hex strings are
    rejected even though they would parse identically. Reason:
    `hashlib.sha256(...).hexdigest()` is always lowercase, so a
    regulator re-canonicalizing a manifest for hash comparison
    would see two distinct byte sequences for the same logical
    digest. Force the producer to .lower() at the boundary."""
    upper = "AA" * 32
    with pytest.raises(ValueError, match="not a 64-char lowercase hex"):
        as_sha256_hex(upper)
    mixed = "Aa" * 32
    with pytest.raises(ValueError, match="not a 64-char lowercase hex"):
        as_sha256_hex(mixed)


def test_v23_emit_manifest_accepts_validated_sha256():
    """v2.3 polish item 3 — emit_manifest's `artifact_sha256`
    parameter is annotated `Optional[Sha256Hex]`. The NewType is
    runtime-equivalent to str so existing plain-str callers still
    work, but going through `as_sha256_hex` is the typed path."""
    import hashlib
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()

    artifact_digest = as_sha256_hex(hashlib.sha256(b"binary").hexdigest())
    m = emit_manifest(tc.functions, artifact_sha256=artifact_digest)
    assert m.artifact_sha256 == artifact_digest
    # Manifest is self-consistent.
    assert verify_manifest_hash(m) is True


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
    assert o.has_enclave_param is True
    # Return is i32 (bare), so enclave_tag is None.
    assert o.enclave_tag is None


def test_v24_proof_manifest_is_frozen():
    """v2.4 item 3 slice 2/3 — emit_manifest returns a frozen
    ProofManifest dataclass; rebinding any attribute raises
    FrozenInstanceError. An attestation manifest must be tamper-
    evident — the producer-side object can no longer be silently
    rewritten between emit and verify."""
    import dataclasses
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    m = emit_manifest(tc.functions)
    assert isinstance(m, ProofManifest)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.manifest_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.helix_version = "tampered"  # type: ignore[misc]


def test_v24_function_obligation_is_frozen():
    """v2.4 item 3 slice 2/3 — the per-function obligation rows are
    frozen too (deep freeze): `m.functions[i].is_pure = ...` — the
    in-memory tamper the Stage-122 plain dict allowed silently — now
    raises FrozenInstanceError."""
    import dataclasses
    src = "@pure fn p() -> i32 { 0 }\nfn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    m = emit_manifest(tc.functions)
    assert isinstance(m.functions, tuple)
    assert all(isinstance(f, FunctionObligation) for f in m.functions)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.functions[0].is_pure = True  # type: ignore[misc]


def test_v24_proof_manifest_to_dict_wire_form():
    """v2.4 item 3 slice 2/3 — `ProofManifest.to_dict()` reproduces
    the Stage-122 wire dict: same keys, `functions` as a list of
    dicts, `effects` as lists. A verifier that re-canonicalizes the
    dict gets the same digest the dataclass stores, and JSON of the
    dataclass equals JSON of its dict (byte-identical)."""
    src = """
    @effect(io)
    fn w() -> i32 { 0 }
    fn main() -> i32 { 0 }
    """
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    m = emit_manifest(tc.functions)
    d = m.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["functions"], list)
    assert all(isinstance(f, dict) for f in d["functions"])
    assert all(isinstance(f["effects"], list) for f in d["functions"])
    assert d["manifest_sha256"] == m.manifest_sha256
    assert verify_manifest_hash(d) is True
    assert serialize_manifest(m) == serialize_manifest(d)


def test_v24_verify_manifest_hash_accepts_dataclass_and_dict():
    """v2.4 item 3 slice 2/3 — verify_manifest_hash accepts both the
    ProofManifest dataclass (producer self-check) and the wire dict
    (the form a verifier gets from json.loads)."""
    src = "fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    tc = TypeChecker(prog)
    tc.check()
    m = emit_manifest(tc.functions)
    assert verify_manifest_hash(m) is True            # dataclass path
    assert verify_manifest_hash(m.to_dict()) is True  # dict path


def test_v25_verify_manifest_hash_dataclass_rich_manifest():
    """v2.5 polish (BE LOW-2 from the end-of-v2.4 5-clean-gate) — pin
    the `isinstance(.., ProofManifest)` branch of verify_manifest_hash
    on a *rich* manifest: multiple functions, a non-empty `effects`
    tuple, and an enclave-tagged return.

    The existing dataclass-path tests
    (`test_v24_verify_manifest_hash_accepts_dataclass_and_dict`,
    `test_stage122_verify_manifest_hash_self_consistency`) use a
    trivial single-`main` module, which barely exercises the deep
    `to_dict()` conversion the dataclass branch relies on — the
    per-function `FunctionObligation.to_dict()` rows and the
    `effects` tuple->list demotion. This feeds the producer dataclass
    straight into verify and asserts the canonical hash survives that
    conversion end-to-end, agreeing byte-for-byte with the dict path
    a real attestation verifier holds."""
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

    m = emit_manifest(tc.functions)
    # Rich-manifest precondition: the dataclass really carries the
    # variety the trivial dataclass-path tests don't.
    assert isinstance(m, ProofManifest)
    assert m.function_count == 4
    assert any(f.effects for f in m.functions)      # @effect(io) -> ("io",)
    assert any(f.enclave_tag for f in m.functions)  # InEnclaveSGX return

    # The branch under test: a ProofManifest dataclass passed directly
    # (producer self-check) — not the json.loads dict a verifier holds.
    assert verify_manifest_hash(m) is True

    # `to_dict()` is the only bridge from the dataclass branch into the
    # canonical-hash path; BE LOW-2 is about trusting it on a manifest
    # with real per-function content. Dataclass and dict paths must
    # agree byte-for-byte.
    assert verify_manifest_hash(m.to_dict()) is True
    assert verify_manifest_hash(m) == verify_manifest_hash(m.to_dict())
