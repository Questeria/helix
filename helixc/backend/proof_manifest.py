"""
helixc/backend/proof_manifest.py — Stage 122 (v2.0 Phase C.3).

Attestation-binding ProofObligation manifest emitter. Walks a
TypeChecker's function-signature table and produces a JSON document
that pairs each function's compiled artifact with its
type-system-derived proof obligations:

- Enclave tags (TyEnclave inner-most: sgx / tz / tdx)
- Effect declarations (@effect labels, including gpu.* sync labels)
- Purity flags
- Module fingerprint (sha256 of canonicalized signatures)

The manifest is signed (placeholder Stage 122 — actual signing
requires a HW-backed key or TEE attestation; substrate ships an
unsigned digest plus a `signature_format` field documenting the
expected signing scheme for downstream tools).

Per v2.0 research Report 3 Layer-3 wedge: "first compiler to bind
information-flow proofs to GPU CC attestation."

Reference attestation flows:
- NVIDIA NRAS (H100 CC attestation verifier)
- Intel SGX EPID / DCAP quote verification
- AMD SEV-SNP attestation report
- Apple Private Cloud Compute code-signing manifest

License: Apache 2.0
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType, Optional, cast


# Manifest format version. Bump when fields are added/removed in a
# non-backward-compatible way.
PROOF_MANIFEST_VERSION = "v2.0-stage122-substrate"


# v2.3 polish item 3 (slice 3 of 3): closed-set signing-scheme tag.
# The manifest's `signature_format` field was a free-form `str`
# (`"ed25519-or-rsa-pss-sha256-DEFERRED"`) — a typo or an
# unrecognized scheme name would slip through to an attestation
# verifier that pivots on the value. A `str`-based Enum keeps the
# JSON wire form a plain string (so existing manifests + the
# canonical hash are unchanged) while giving static checkers a
# closed set and runtime code a `SignatureFormat(value)` parse that
# rejects unknown schemes at the boundary.
#
# Semantics:
#   DEFERRED:        Stage 122 substrate — manifest is emitted
#                    UNSIGNED; the format string documents the
#                    *expected* scheme for downstream signing tools.
#   ED25519:         Ed25519 detached signature over the canonical
#                    manifest bytes (compact, fast verify).
#   RSA_PSS_SHA256:  RSA-PSS / SHA-256 — for HSM / TPM key stores
#                    that don't expose Ed25519.
#   TEE_QUOTE:       signature is a TEE attestation quote (SGX DCAP /
#                    SEV-SNP report / NRAS bundle) — verified against
#                    the platform's attestation service, not a bare
#                    public key.
class SignatureFormat(str, Enum):
    """Closed set of manifest signing schemes (v2.3 item 3 slice 3)."""
    DEFERRED = "ed25519-or-rsa-pss-sha256-DEFERRED"
    ED25519 = "ed25519"
    RSA_PSS_SHA256 = "rsa-pss-sha256"
    TEE_QUOTE = "tee-quote"


# v2.3 polish item 3 (slice 1 of 3): typed marker for sha256 hex
# digests. NewType is runtime-equivalent to str, so existing callers
# passing plain str continue to work — this tightens static-typing
# annotations and gives the validator (`as_sha256_hex`) a distinct
# return type. Downstream attestation code that wants to enforce
# "this str was checked" at the type level can require Sha256Hex.
Sha256Hex = NewType("Sha256Hex", str)


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def as_sha256_hex(value: Any) -> Sha256Hex:
    """v2.3 polish item 3 — validate-and-tag a sha256 hex digest.

    Returns the input typed as `Sha256Hex` if it is exactly 64
    lowercase hex characters. Raises ValueError otherwise so a
    producer that hands manifest emitters a malformed digest fails
    at the boundary, not deep inside attestation verification.

    Accepts only lowercase to match `hashlib.sha256().hexdigest()`
    output — mixed-case inputs are rejected (regulators that
    re-canonicalize manifests for hash comparison would otherwise
    see two distinct byte sequences for the same digest).
    """
    if not isinstance(value, str):
        raise ValueError(
            f"as_sha256_hex: expected str, got {type(value).__name__}: "
            f"{value!r}. sha256 hex digests must be strings."
        )
    if not _SHA256_HEX_RE.match(value):
        raise ValueError(
            f"as_sha256_hex: {value!r} is not a 64-char lowercase hex "
            f"sha256 digest. Expected exactly 64 chars matching "
            f"[0-9a-f]; got len={len(value)}. Mixed-case digests are "
            f"rejected (canonicalization compatibility); lowercase "
            f"the input via .lower() if you trust the source."
        )
    return cast(Sha256Hex, value)


def _canonicalize_sig(sig_dict: dict) -> str:
    """Stable string representation of a function signature for hashing.

    Uses sorted-key JSON so two manifests emitted in different orders
    produce the same hash. The hash is what an attestation verifier
    compares against the runtime-loaded manifest.
    """
    return json.dumps(sig_dict, sort_keys=True, separators=(",", ":"))


def extract_enclave_tag(ty: Any) -> Optional[str]:
    """Stage 122 — walk a Type chain to find the innermost TyEnclave
    tag. Returns the enclave name (e.g. "sgx") or None if untagged.

    Mirrors the `_find_enclave_name` helper from typecheck.py:5641
    but lives in the backend so the manifest emitter can be invoked
    without re-importing the typechecker (avoids circular deps).
    """
    # Avoid the import cycle: typecheck imports backend (via grad_pass
    # cascading) so backend cannot import typecheck. Walk via duck-
    # typing on the `enclave` attribute pattern.
    #
    # v2.2 polish item 12 (BE LOW-1 from v2.1 5-clean-gate): the prior
    # `seen < 32` loop bound silently returned None on pathological
    # type chains exceeding depth 32 — manifest would then say "no
    # enclave tag" when one existed deep in the chain. Phase-0 wouldn't
    # hit this in practice but the silent miscategorization was a
    # latent bug. R1 fix: raise ValueError with full path-trace info
    # when depth exhausted, so an untagged-vs-too-deep return is
    # distinguishable from the deferred path.
    MAX_DEPTH = 32
    seen = 0
    while ty is not None:
        if hasattr(ty, "enclave") and hasattr(ty, "inner"):
            return ty.enclave
        if hasattr(ty, "inner"):
            if seen >= MAX_DEPTH:
                raise ValueError(
                    f"extract_enclave_tag: type-chain depth exceeded "
                    f"{MAX_DEPTH} levels without finding an enclave-"
                    f"typed or terminal node. Type at depth-cap: "
                    f"{type(ty).__name__}. This likely indicates a "
                    f"cyclic .inner chain or a pathologically deep "
                    f"nested type — please file a bug with the "
                    f"producing function name."
                )
            ty = ty.inner
            seen += 1
        else:
            return None
    return None


@dataclass(frozen=True)
class FunctionObligation:
    """v2.4 item 3 slice 2/3 — frozen per-function proof-obligation row.

    Stage 122 emitted these as plain mutable dicts. An attestation
    manifest is an integrity artifact, so the obligation rows are now
    immutable too: `manifest.functions[i].is_pure = ...` (an in-memory
    tamper the Stage-122 dict allowed silently) raises
    FrozenInstanceError.

    `effects` is a tuple (not a list) so the record is hashable and
    deeply immutable. `to_dict()` reproduces the exact Stage-122 wire
    dict (effects back to a sorted list) so the canonical manifest
    hash stays byte-identical to pre-dataclass manifests.

    `has_enclave_param` tracks whether any parameter carries an
    enclave tag — downstream audit tooling treats enclave-typed
    params separately from enclave-typed returns (`enclave_tag`).
    """
    name: str
    is_pure: bool
    effects: tuple[str, ...]
    enclave_tag: Optional[str]
    param_count: int
    has_enclave_param: bool

    def to_dict(self) -> dict[str, Any]:
        """Stage-122-compatible wire dict (effects as a list)."""
        return {
            "name": self.name,
            "is_pure": self.is_pure,
            "effects": list(self.effects),
            "enclave_tag": self.enclave_tag,
            "param_count": self.param_count,
            "has_enclave_param": self.has_enclave_param,
        }


def fn_proof_obligations(sig: Any) -> FunctionObligation:
    """Stage 122 — extract proof obligations for one function signature.

    Reads from FunctionSig (typecheck.py FunctionSig dataclass).

    v2.4 item 3 slice 2/3: returns a frozen `FunctionObligation`
    dataclass (was a plain dict). Callers use attribute access
    (`o.is_pure`) not item access (`o["is_pure"]`); `o.to_dict()`
    recovers the Stage-122 wire form.
    """
    # Check params for enclave tags.
    has_enclave_param = False
    for (_pname, pty) in sig.params:
        if extract_enclave_tag(pty) is not None:
            has_enclave_param = True
            break
    return FunctionObligation(
        name=sig.name,
        is_pure=bool(sig.is_pure),
        effects=tuple(sorted(sig.effects)),
        enclave_tag=extract_enclave_tag(sig.ret),
        param_count=len(sig.params),
        has_enclave_param=has_enclave_param,
    )


@dataclass(frozen=True)
class ProofManifest:
    """v2.4 item 3 slice 2/3 — frozen attestation-binding manifest.

    Stage 122 emitted the manifest as a plain mutable dict. An
    attestation manifest's whole job is to be tamper-evident, yet a
    mutable dict let any code path silently rewrite a proof
    obligation between emit and verify. The producer side now gets a
    frozen dataclass: attribute rebinding raises FrozenInstanceError,
    and `functions` is a tuple of frozen `FunctionObligation` rows so
    the freeze is deep.

    The consumer side is deliberately unchanged: `verify_manifest_hash`
    and `serialize_manifest` still accept the plain dict an attestation
    verifier gets from `json.loads` — untrusted JSON never arrives as a
    dataclass. `to_dict()` bridges producer dataclass → wire dict; the
    canonical `manifest_sha256` is byte-identical to pre-dataclass
    manifests (same field set, `functions`/`effects` back to lists).
    """
    format_version: str
    helix_version: str
    artifact_path: Optional[str]
    artifact_sha256: Optional[Sha256Hex]
    function_count: int
    functions: tuple[FunctionObligation, ...]
    signature_format: str
    signature: Optional[str]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Stage-122-compatible wire dict — the form an attestation
        verifier re-canonicalizes. Field set + values are byte-
        identical to the pre-dataclass manifest dict."""
        return {
            "format_version": self.format_version,
            "helix_version": self.helix_version,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "function_count": self.function_count,
            "functions": [f.to_dict() for f in self.functions],
            "signature_format": self.signature_format,
            "signature": self.signature,
            "manifest_sha256": self.manifest_sha256,
        }


def emit_manifest(
    functions: dict[str, Any],
    artifact_path: Optional[str] = None,
    artifact_sha256: Optional[Sha256Hex] = None,
    helix_version: str = "v2.0-substrate",
    signature_format: SignatureFormat = SignatureFormat.DEFERRED,
) -> ProofManifest:
    """Stage 122 — emit a ProofObligation manifest.

    Args:
        functions: typecheck.TypeChecker.functions dict
                   (name -> FunctionSig).
        artifact_path: optional path to the compiled ELF/PTX for
                       this manifest (logged but not required —
                       allows manifest-only emit for CI verification).
        artifact_sha256: optional sha256 of the compiled artifact;
                         if provided, bound into the manifest hash.
        helix_version: compiler version string for downstream
                       attestation gates that pin Helix versions.
        signature_format: v2.3 item 3 slice 3 — the signing scheme
                          this manifest will be signed under. Defaults
                          to SignatureFormat.DEFERRED (Stage 122
                          substrate emits unsigned). A plain str is
                          accepted and coerced via SignatureFormat(...)
                          which rejects unknown schemes loudly.

    Returns:
        A frozen `ProofManifest` dataclass (v2.4 item 3 slice 2/3 —
        was a plain dict). Caller serializes via `serialize_manifest`
        + signs externally (HW-backed key or TEE quote).

    The manifest's `manifest_sha256` field is computed over the
    sorted-keys canonical form of every other field except
    `signature` — this is the hash an attestation verifier
    challenges.

    Raises:
        ValueError: if `signature_format` is a str that is not a
            recognized SignatureFormat value.
    """
    # v2.3 item 3 slice 3: coerce a str arg through the Enum so an
    # unrecognized scheme name fails at the boundary, not inside a
    # downstream attestation verifier. Enum members pass through
    # unchanged; valid str values are accepted; bad values raise.
    if not isinstance(signature_format, SignatureFormat):
        try:
            signature_format = SignatureFormat(signature_format)
        except ValueError as exc:
            valid = ", ".join(repr(s.value) for s in SignatureFormat)
            raise ValueError(
                f"emit_manifest: signature_format="
                f"{signature_format!r} is not a recognized signing "
                f"scheme. Valid values: {valid}."
            ) from exc

    fn_obligations = tuple(
        fn_proof_obligations(functions[name])
        for name in sorted(functions.keys())
    )

    # Canonical-hash input: the wire form minus `signature` (Stage 122
    # excludes it) and minus `manifest_sha256` (which doesn't exist
    # yet). Field set + values are byte-identical to the Stage-122
    # dict the hash was first defined over — `to_dict()` on each
    # FunctionObligation reproduces the original per-function dict.
    # Signature plumbing — Stage 122 ships unsigned + format
    # declaration. Stage 123+ may add HW-backed signing once the GPU
    # CI substrate (Stage 129) provides attestation endpoints. v2.3
    # item 3 slice 3: `signature_format` stores the Enum's `.value`
    # (plain str) so the wire form + hash are byte-identical to
    # pre-Enum manifests.
    hash_body: dict[str, Any] = {
        "format_version": PROOF_MANIFEST_VERSION,
        "helix_version": helix_version,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "function_count": len(fn_obligations),
        "functions": [f.to_dict() for f in fn_obligations],
        "signature_format": signature_format.value,
    }
    manifest_sha256 = hashlib.sha256(
        _canonicalize_sig(hash_body).encode("utf-8")
    ).hexdigest()

    return ProofManifest(
        format_version=PROOF_MANIFEST_VERSION,
        helix_version=helix_version,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        function_count=len(fn_obligations),
        functions=fn_obligations,
        signature_format=signature_format.value,
        signature=None,
        manifest_sha256=manifest_sha256,
    )


def serialize_manifest(
    manifest: ProofManifest | dict, indent: Optional[int] = 2
) -> str:
    """Stage 122 — serialize a manifest to canonical JSON.

    `indent=2` (default) produces human-readable output suitable for
    audit/HIPAA-EU-AI-Act inspection. Pass `indent=None` for the
    minimal-bytes form an attestation verifier would re-canonicalize.

    Keys are sorted so two emitters running in parallel on the same
    module produce byte-identical output (regression-test friendly).

    v2.4 item 3 slice 2/3: accepts a `ProofManifest` dataclass
    (producer side) or a plain dict (a verifier re-serializing a
    deserialized manifest). The dataclass is converted to its wire
    dict first; output bytes are identical either way.
    """
    if isinstance(manifest, ProofManifest):
        manifest = manifest.to_dict()
    return json.dumps(manifest, sort_keys=True, indent=indent,
                      separators=(",", ": ") if indent else (",", ":"))


def verify_manifest_hash(manifest: ProofManifest | dict) -> bool:
    """Stage 122 — verify that `manifest["manifest_sha256"]` matches the
    canonical hash of the rest of the manifest. Returns True if the
    hash is consistent (verifier-ready), False otherwise.

    v2.4 item 3 slice 2/3: also accepts a `ProofManifest` dataclass
    (producer-side self-check) — it is converted to its wire dict
    first. The dict path is unchanged: a real attestation verifier
    deserializes untrusted JSON into a dict, never a dataclass, so
    all the malformed-input guards below still apply to that dict.

    v2.2 polish item 3 (BE MED-3 from v2.1 5-clean-gate): the prior
    code collapsed two distinct outcomes:
      (a) `manifest_sha256` field missing → return False
      (b) field present but doesn't match canonical hash → return False
    Attestation verifiers downstream couldn't distinguish a malformed
    manifest from a tampered one. R1 fix: case (a) now raises ValueError
    with a clear "missing field" diagnostic; False is reserved for
    case (b) — actual hash mismatch (tamper signal).

    A real attestation flow would also verify `manifest["signature"]`
    against a HW-backed public key. Stage 122 substrate ships the
    hash-consistency check; signature verification lands later.
    """
    # v2.2 5-clean-gate BE LOW-1 audit-fix: non-dict input previously
    # raised a generic `TypeError: argument of type 'NoneType' is not
    # iterable` from the `not in` membership test below. Docstring
    # promises ValueError-with-diagnostic for malformed input —
    # surface that contract here so non-dict inputs get the same
    # error class as the other malformed-manifest paths.
    if isinstance(manifest, ProofManifest):
        manifest = manifest.to_dict()
    if not isinstance(manifest, dict):
        raise ValueError(
            f"verify_manifest_hash: expected dict or ProofManifest, got "
            f"{type(manifest).__name__}: {manifest!r}. Manifest is "
            f"malformed; reject before attestation verification."
        )
    if "manifest_sha256" not in manifest:
        raise ValueError(
            "verify_manifest_hash: manifest missing required "
            "`manifest_sha256` field. The manifest is malformed; "
            "reject before attestation verification (this is NOT a "
            "hash-mismatch signal — use try/except around this call "
            "to distinguish 'malformed' from 'tampered')."
        )
    claimed = manifest["manifest_sha256"]
    # v2.2 5-clean-gate BE MEDIUM-1 audit-fix: non-string `claimed`
    # values (int, list, dict) previously fell through to the
    # `claimed == computed` check at the bottom and returned False —
    # collapsing "wrong-type manifest_sha256" with "actual hash
    # mismatch (tamper)". R1 type-guards close that gap.
    if not isinstance(claimed, str):
        raise ValueError(
            f"verify_manifest_hash: `manifest_sha256` must be a str, "
            f"got {type(claimed).__name__}: {claimed!r}. Manifest is "
            f"malformed; reject before attestation."
        )
    if not claimed:
        raise ValueError(
            "verify_manifest_hash: `manifest_sha256` field is empty "
            "or falsy. The manifest is malformed; treat as `not "
            "tampered, also not verified` and reject."
        )
    canonical = _canonicalize_sig({
        k: v for k, v in manifest.items()
        if k not in ("signature", "manifest_sha256")
    })
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return claimed == computed
