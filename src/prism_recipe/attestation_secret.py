"""Build-time per-run attestation secret (mechanism 6).

BASE injects a unique value at image build via BuildKit secret mounts
(``--secret id=attestation_secret`` / ``RUN --mount=type=secret``). The sidecar
reads the baked file when answering challenges.

**Honesty (mech 6):** a miner with root on the pod can extract this file. The
mechanism raises the cost of forging an attestation response; it does **not**
create a hardware root of trust and is never sufficient for elevation alone.
This is tamper-evidence, not tamper-prevention.

Never put the secret in ARG, ENV, or LABEL — those bake into ``docker history``
and image config. Only the destination file content is committed to a layer;
the BuildKit mount itself is not stored in layer metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# BuildKit --secret id=... / RUN --mount=type=secret,id=...
BUILDKIT_SECRET_ID: Final[str] = "attestation_secret"

# Baked destination inside the image (mode 0400). Sidecar reads this path.
ATTESTATION_HMAC_KEY_PATH: Final[str] = "/run/prism/attestation_hmac_key"
ATTESTATION_HMAC_KEY_MODE: Final[int] = 0o400

# Paths omitted from hermetic rootfs content digests so two builds of the same
# commit can carry different secrets while code layers still MATCH.
ROOTFS_DIGEST_EXCLUDE_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/run/prism/attestation_hmac_key",
        "run/prism/attestation_hmac_key",
        "./run/prism/attestation_hmac_key",
    }
)


class AttestationSecretError(RuntimeError):
    """Fail-closed attestation secret read failure."""


def is_rootfs_excluded_path(name: str) -> bool:
    """Return True if ``name`` (tar member path) must be ignored for code MATCH."""
    normalized = name.strip()
    if normalized in ROOTFS_DIGEST_EXCLUDE_PATHS:
        return True
    # Strip leading ./ and leading /
    stripped = normalized.lstrip("./")
    return stripped == "run/prism/attestation_hmac_key" or normalized.endswith(
        "/run/prism/attestation_hmac_key"
    )


def read_attestation_hmac_key(path: str | Path | None = None) -> bytes:
    """Read the baked per-build secret bytes.

    Raises:
        AttestationSecretError: missing, empty, or unreadable key file.
    """
    key_path = Path(path if path is not None else ATTESTATION_HMAC_KEY_PATH)
    if not key_path.is_file():
        raise AttestationSecretError(
            f"attestation hmac key missing (not found): {key_path}"
        )
    data = key_path.read_bytes()
    if not data:
        raise AttestationSecretError(f"attestation hmac key empty: {key_path}")
    return data
