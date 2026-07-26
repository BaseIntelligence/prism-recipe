"""Wire encoding for signed attestation answers (BASE-facing JSON)."""

from __future__ import annotations

from typing import Any, Literal

from prism_recipe.attestation.payload import (
    ALGORITHM,
    SCHEMA_VERSION,
    VALID_VARIANTS,
    AttestationPayload,
    SignedAttestation,
)


def signed_attestation_to_wire(signed: SignedAttestation) -> dict[str, Any]:
    """Serialize a signed attestation for BASE HTTP consume.

    Includes honesty flags outside the signed body so dumps cannot be mistaken
    for hardware attestation or tier elevation.
    """
    payload = signed.payload
    body = {
        "schema_version": signed.schema_version,
        "algorithm": signed.algorithm,
        "signature": signed.signature,
        "payload": {
            "nonce": payload.nonce,
            "digest": payload.digest,
            "pod_id": payload.pod_id,
            "variant": payload.variant,
            "sealed_manifest_hashes": dict(payload.sealed_manifest_hashes),
            "build_secret_response": payload.build_secret_response,
        },
        # Not covered by HMAC — advisory honesty markers for operators/evidence.
        "hardware_root_of_trust": False,
        "sufficient_alone_for_tier_elevation": False,
        "proves": "entity_holding_in_image_secret_responded",
    }
    # Keep schema constants aligned with payload module.
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["algorithm"] == ALGORITHM
    return body


def signed_from_wire(wire: dict[str, Any]) -> SignedAttestation:
    """Parse a wire dict back into ``SignedAttestation`` (tests / BASE mirror)."""
    raw_payload = wire["payload"]
    if not isinstance(raw_payload, dict):
        msg = "wire.payload must be an object"
        raise ValueError(msg)
    hashes = raw_payload["sealed_manifest_hashes"]
    if not isinstance(hashes, dict):
        msg = "wire.payload.sealed_manifest_hashes must be an object"
        raise ValueError(msg)
    variant_raw = str(raw_payload["variant"]).strip().lower()
    if variant_raw not in VALID_VARIANTS:
        msg = f"invalid variant in wire: {variant_raw!r}"
        raise ValueError(msg)
    variant: Literal["cpu", "cuda"] = "cpu" if variant_raw == "cpu" else "cuda"
    payload = AttestationPayload(
        nonce=str(raw_payload["nonce"]),
        digest=str(raw_payload["digest"]),
        pod_id=str(raw_payload["pod_id"]),
        variant=variant,
        sealed_manifest_hashes={str(k): str(v) for k, v in hashes.items()},
        build_secret_response=str(raw_payload["build_secret_response"]),
    )
    return SignedAttestation(
        payload=payload,
        signature=str(wire["signature"]),
        algorithm=str(wire.get("algorithm", ALGORITHM)),
        schema_version=str(wire.get("schema_version", SCHEMA_VERSION)),
    )
