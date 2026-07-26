"""TDD: build-time per-run attestation secret (mech 6).

BASE injects a unique secret via BuildKit secret mounts. The value must never
appear as ARG/ENV/LABEL (those bake into history/layer metadata). Sidecar reads
the baked file at the canonical path. Root miners can extract it — raises cost,
not trust.
"""

from __future__ import annotations

import importlib.util
import io
import re
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = (ROOT / "Dockerfile", ROOT / "Dockerfile.cuda")
ROOTFS_HELPER = ROOT / "scripts" / "rootfs_content_digest.py"

# Names that must never be ARG/ENV/LABEL (would leak into history / config).
_FORBIDDEN_SECRET_NAMES = (
    "ATTESTATION_SECRET",
    "ATTESTATION_HMAC_KEY",
    "PRISM_ATTESTATION_SECRET",
    "BUILD_SECRET",
    "attestation_secret",
)

_SECRET_MOUNT_RE = re.compile(
    r"RUN\s+--mount=type=secret,[^\n]*\bid=attestation_secret\b",
    re.IGNORECASE | re.MULTILINE,
)
_ARG_ENV_LABEL_RE = re.compile(
    r"^\s*(ARG|ENV|LABEL)\s+.*\b("
    + "|".join(re.escape(n) for n in _FORBIDDEN_SECRET_NAMES)
    + r")\b",
    re.IGNORECASE | re.MULTILINE,
)


def _load_attestation_secret():
    from prism_recipe import attestation_secret as mod

    return mod


def _load_rootfs_helper():
    assert ROOTFS_HELPER.is_file(), f"missing {ROOTFS_HELPER}"
    spec = importlib.util.spec_from_file_location("rootfs_content_digest", ROOTFS_HELPER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_secret_path_convention_is_stable() -> None:
    """Given: mech 6 path contract. When: import constants. Then: fixed path + mode + secret id."""
    mod = _load_attestation_secret()
    assert mod.BUILDKIT_SECRET_ID == "attestation_secret"
    assert mod.ATTESTATION_HMAC_KEY_PATH == "/run/prism/attestation_hmac_key"
    assert mod.ATTESTATION_HMAC_KEY_MODE == 0o400
    # Rootfs hermetic compare must ignore this path so code MATCH survives unique secrets.
    assert "/run/prism/attestation_hmac_key" in mod.ROOTFS_DIGEST_EXCLUDE_PATHS
    assert "run/prism/attestation_hmac_key" in mod.ROOTFS_DIGEST_EXCLUDE_PATHS


def test_read_attestation_hmac_key_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy: sidecar reader returns bytes from the canonical path when present."""
    mod = _load_attestation_secret()
    key_path = tmp_path / "attestation_hmac_key"
    key_path.write_bytes(b"per-build-secret-value-aa")
    key_path.chmod(0o400)
    monkeypatch.setattr(mod, "ATTESTATION_HMAC_KEY_PATH", str(key_path))
    assert mod.read_attestation_hmac_key() == b"per-build-secret-value-aa"


def test_read_attestation_hmac_key_missing_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edge: missing key file raises typed error (sidecar must not invent a secret)."""
    mod = _load_attestation_secret()
    missing = tmp_path / "nope"
    monkeypatch.setattr(mod, "ATTESTATION_HMAC_KEY_PATH", str(missing))
    with pytest.raises(mod.AttestationSecretError, match="missing|not found|absent"):
        mod.read_attestation_hmac_key()


def test_read_attestation_hmac_key_empty_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edge: empty key file is rejected (fail closed)."""
    mod = _load_attestation_secret()
    key_path = tmp_path / "empty"
    key_path.write_bytes(b"")
    monkeypatch.setattr(mod, "ATTESTATION_HMAC_KEY_PATH", str(key_path))
    with pytest.raises(mod.AttestationSecretError, match="empty"):
        mod.read_attestation_hmac_key()


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_dockerfile_uses_buildkit_secret_mount(dockerfile: Path) -> None:
    """Given: image build. When: parse Dockerfile. Then: secret mount id=attestation_secret."""
    assert dockerfile.is_file(), f"missing {dockerfile}"
    text = dockerfile.read_text(encoding="utf-8")
    joined = text.replace("\\\n", " ")
    assert _SECRET_MOUNT_RE.search(joined), (
        f"{dockerfile.name}: must inject via RUN --mount=type=secret,id=attestation_secret "
        "(not ARG/ENV)"
    )
    assert "/run/prism/attestation_hmac_key" in text, (
        f"{dockerfile.name}: must write secret to /run/prism/attestation_hmac_key"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_dockerfile_forbids_arg_env_label_of_secret_name(dockerfile: Path) -> None:
    """Given: leak-prevention policy. When: scan ARG/ENV/LABEL. Then: no secret name tokens."""
    text = dockerfile.read_text(encoding="utf-8")
    match = _ARG_ENV_LABEL_RE.search(text)
    assert match is None, (
        f"{dockerfile.name}: secret must not appear as ARG/ENV/LABEL "
        f"(leaks to history/config): {match.group(0)!r}"
    )


def test_leaky_env_dockerfile_fixture_is_rejected() -> None:
    """RED characterization: a Dockerfile that ENV-bakes the secret must fail the forbid scan.

    Evidence target for 10-red-secret-leak.txt — proves the gate catches ENV leaks.
    """
    leaky = (
        "FROM python:3.12-slim-bookworm@sha256:"
        + ("a" * 64)
        + "\n"
        "ENV ATTESTATION_SECRET=super-secret-value-should-not-bake\n"
        "RUN echo done\n"
    )
    match = _ARG_ENV_LABEL_RE.search(leaky)
    assert match is not None, "gate must detect ENV ATTESTATION_SECRET as a leak"
    assert "ATTESTATION_SECRET" in match.group(0)


def test_rootfs_digest_excludes_attestation_secret_path() -> None:
    """Adjacent: two rootfs tars that differ only in the hmac key still MATCH when excluded."""
    mod = _load_rootfs_helper()
    assert hasattr(mod, "is_excluded_path") or hasattr(mod, "content_digest_from_tar")

    def _tar_with(secret: bytes) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            # Code file identical in both builds
            data = b"print('hermetic')\n"
            info = tarfile.TarInfo(name="./app/src/prism_recipe/harness.py")
            info.size = len(data)
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))
            # Per-build secret (intentionally different)
            sinfo = tarfile.TarInfo(name="./run/prism/attestation_hmac_key")
            sinfo.size = len(secret)
            sinfo.mode = 0o400
            tf.addfile(sinfo, io.BytesIO(secret))
        return buf.getvalue()

    d1 = mod.content_digest_from_bytes(_tar_with(b"secret-build-AAAA"))
    d2 = mod.content_digest_from_bytes(_tar_with(b"secret-build-BBBB"))
    assert d1 == d2, "rootfs digest must exclude attestation hmac key so code MATCH holds"

    # Control: differing code must still DIFF
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"print('mutated')\n"
        info = tarfile.TarInfo(name="./app/src/prism_recipe/harness.py")
        info.size = len(data)
        info.mode = 0o644
        tf.addfile(info, io.BytesIO(data))
    d3 = mod.content_digest_from_bytes(buf.getvalue())
    assert d3 != d1


def test_module_docstring_states_root_miner_can_extract() -> None:
    """Honesty: mech 6 raises cost, not trust — docstring must say root can extract."""
    mod = _load_attestation_secret()
    doc = (mod.__doc__ or "").lower()
    assert "root" in doc
    assert "extract" in doc or "read" in doc
    assert "trust" in doc or "tamper-evidence" in doc or "cost" in doc
