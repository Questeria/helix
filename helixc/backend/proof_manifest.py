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
from typing import Any, Optional


# Manifest format version. Bump when fields are added/removed in a
# non-backward-compatible way.
PROOF_MANIFEST_VERSION = "v2.0-stage122-substrate"


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


def fn_proof_obligations(sig: Any) -> dict:
    """Stage 122 — extract proof obligations for one function signature.

    Reads from FunctionSig (typecheck.py FunctionSig dataclass).
    Returns a JSON-serializable dict with:
      name, is_pure, effects (sorted list), enclave_tag (or null),
      param_count, has_enclave_param (bool — for downstream audit
      tooling that wants to track enclave-typed params separately
      from enclave-typed returns).
    """
    obligations: dict[str, Any] = {
        "name": sig.name,
        "is_pure": bool(sig.is_pure),
        "effects": sorted(sig.effects),
        "enclave_tag": extract_enclave_tag(sig.ret),
        "param_count": len(sig.params),
    }
    # Check params for enclave tags.
    has_enclave_param = False
    for (_pname, pty) in sig.params:
        if extract_enclave_tag(pty) is not None:
            has_enclave_param = True
            break
    obligations["has_enclave_param"] = has_enclave_param
    return obligations


def emit_manifest(
    functions: dict[str, Any],
    artifact_path: Optional[str] = None,
    artifact_sha256: Optional[str] = None,
    helix_version: str = "v2.0-substrate",
) -> dict:
    """Stage 122 — emit a ProofObligation manifest dict.

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

    Returns:
        A dict containing the manifest. Caller serializes via
        json.dumps + signs externally (HW-backed key or TEE quote).

    The returned dict includes a `manifest_sha256` field over the
    sorted-keys canonical form of every other field — this is the
    hash an attestation verifier challenges.
    """
    fn_obligations = []
    for name in sorted(functions.keys()):
        sig = functions[name]
        fn_obligations.append(fn_proof_obligations(sig))

    manifest: dict[str, Any] = {
        "format_version": PROOF_MANIFEST_VERSION,
        "helix_version": helix_version,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "function_count": len(fn_obligations),
        "functions": fn_obligations,
        # Signature plumbing — Stage 122 ships unsigned + format
        # declaration. Stage 123+ may add HW-backed signing once
        # the GPU CI substrate (Stage 129) provides attestation
        # endpoints.
        "signature_format": "ed25519-or-rsa-pss-sha256-DEFERRED",
        "signature": None,
    }

    # Compute the canonical hash over everything-except-signature.
    # Verifier re-canonicalizes and compares.
    canonical = _canonicalize_sig({
        k: v for k, v in manifest.items()
        if k not in ("signature", "manifest_sha256")
    })
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return manifest


def serialize_manifest(manifest: dict, indent: Optional[int] = 2) -> str:
    """Stage 122 — serialize a manifest dict to canonical JSON.

    `indent=2` (default) produces human-readable output suitable for
    audit/HIPAA-EU-AI-Act inspection. Pass `indent=None` for the
    minimal-bytes form an attestation verifier would re-canonicalize.

    Keys are sorted so two emitters running in parallel on the same
    module produce byte-identical output (regression-test friendly).
    """
    return json.dumps(manifest, sort_keys=True, indent=indent,
                      separators=(",", ": ") if indent else (",", ":"))


def verify_manifest_hash(manifest: dict) -> bool:
    """Stage 122 — verify that `manifest["manifest_sha256"]` matches the
    canonical hash of the rest of the manifest. Returns True if the
    hash is consistent (verifier-ready), False otherwise.

    A real attestation flow would also verify `manifest["signature"]`
    against a HW-backed public key. Stage 122 substrate ships the
    hash-consistency check; signature verification lands later.
    """
    claimed = manifest.get("manifest_sha256")
    if not claimed:
        return False
    canonical = _canonicalize_sig({
        k: v for k, v in manifest.items()
        if k not in ("signature", "manifest_sha256")
    })
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return claimed == computed
