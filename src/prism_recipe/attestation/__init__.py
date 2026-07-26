"""Attestation payload schema + HMAC verify (mirror of base.attestation).

Canonical protocol lives in both repos with identical test vectors because
prism-recipe cannot import base inside the miner image. Keep payloads in lockstep.
"""

from prism_recipe.attestation.payload import (
    ALGORITHM,
    SCHEMA_VERSION,
    VALID_VARIANTS,
    AttestationPayload,
    AttestationPayloadError,
    AttestationVerifyError,
    AttestationVerifyReason,
    AttestationVerifyResult,
    SignedAttestation,
    canonical_payload_bytes,
    compute_build_secret_response,
    derive_attestation_key,
    sign_attestation_payload,
    verify_attestation_payload,
)

__all__ = [
    "ALGORITHM",
    "SCHEMA_VERSION",
    "AttestationPayload",
    "AttestationPayloadError",
    "AttestationVerifyError",
    "AttestationVerifyReason",
    "AttestationVerifyResult",
    "SignedAttestation",
    "VALID_VARIANTS",
    "canonical_payload_bytes",
    "compute_build_secret_response",
    "derive_attestation_key",
    "sign_attestation_payload",
    "verify_attestation_payload",
]
