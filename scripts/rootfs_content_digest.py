#!/usr/bin/env python3
"""Canonical rootfs content digest from `docker export` stream on stdin.

Hashes sorted (path, kind, mode, content-sha256) tuples so the digest is stable
even when tar member order or mtimes differ. Used to prove byte-identical image
filesystem contents across two clean rebuilds.

**Mech 6:** the per-build attestation hmac key path is excluded so two builds of
the same commit can carry different secrets while code rootfs still MATCH.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from typing import BinaryIO, Final

# Keep in sync with prism_recipe.attestation_secret.ROOTFS_DIGEST_EXCLUDE_PATHS
# (script must stay import-light for bare `python3` in CI without package install).
_EXCLUDED_SUFFIXES: Final[tuple[str, ...]] = (
    "run/prism/attestation_hmac_key",
)


def is_excluded_path(name: str) -> bool:
    """Return True if tar member path is the per-build attestation secret."""
    normalized = name.strip().lstrip("./")
    if normalized in _EXCLUDED_SUFFIXES:
        return True
    return any(normalized.endswith(suf) or normalized == suf for suf in _EXCLUDED_SUFFIXES)


def content_digest_from_tar(tf: tarfile.TarFile) -> str:
    """Compute canonical sha256 hex digest from an open tar stream."""
    h = hashlib.sha256()
    entries: list[tuple[str, bytes]] = []
    for member in tf:
        name = member.name
        if name in {".", "./"}:
            continue
        if is_excluded_path(name):
            continue
        if member.issym() or member.islnk():
            payload = f"LINK\0{name}\0{member.linkname}\0{member.mode:o}".encode()
            entries.append((name, payload))
            continue
        if member.isdir():
            payload = f"DIR\0{name}\0{member.mode:o}".encode()
            entries.append((name, payload))
            continue
        if not member.isreg():
            payload = f"SPECIAL\0{name}\0{member.mode:o}\0{member.type}".encode()
            entries.append((name, payload))
            continue
        handle = tf.extractfile(member)
        data = handle.read() if handle is not None else b""
        file_h = hashlib.sha256(data).hexdigest()
        payload = f"FILE\0{name}\0{member.mode:o}\0{member.size}\0{file_h}".encode()
        entries.append((name, payload))
    for _name, payload in sorted(entries, key=lambda item: item[0].encode()):
        h.update(payload)
        h.update(b"\n")
    return h.hexdigest()


def content_digest_from_bytes(blob: bytes) -> str:
    """Compute digest from a complete tar archive in memory (unit tests)."""
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tf:
        return content_digest_from_tar(tf)


def content_digest_from_stream(stream: BinaryIO) -> str:
    """Compute digest from a streaming tar (docker export pipe)."""
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        return content_digest_from_tar(tf)


def main() -> int:
    digest = content_digest_from_stream(sys.stdin.buffer)
    sys.stdout.write(digest + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
