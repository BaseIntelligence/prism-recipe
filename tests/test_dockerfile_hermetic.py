"""Hermetic Dockerfile gate: digest-pinned bases, hashed deps, no editable installs.

Parses Dockerfile and Dockerfile.cuda. Fails closed on floating tags, unhashed
installs, editable (-e) image-path installs, or missing SOURCE_DATE_EPOCH.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = (ROOT / "Dockerfile", ROOT / "Dockerfile.cuda")
LOCKFILE = ROOT / "requirements.lock"

_FROM_RE = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)", re.MULTILINE | re.IGNORECASE)
_ARG_RE = re.compile(
    r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$",
    re.MULTILINE,
)
_SHA256_SUFFIX = re.compile(r"@sha256:[a-fA-F0-9]{64}$")
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_EDITABLE_PIP = re.compile(
    r"pip(?:3)?\s+install\b[^\n]*?(?:\s-e\s|\s--editable\s)",
    re.IGNORECASE,
)
_HASH_FLAG = re.compile(r"--require-hashes|--hash=sha256:", re.IGNORECASE)


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line.strip()
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].strip()
    return line.strip()


def _arg_defaults(text: str) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for match in _ARG_RE.finditer(text):
        name = match.group(1)
        raw = _strip_comment(match.group(2)).strip().strip("\"'")
        defaults[name] = raw
    return defaults


def _expand_vars(value: str, defaults: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in defaults:
            return match.group(0)
        return defaults[key]

    prev = None
    cur = value
    # Bound expansion to avoid cycles.
    for _ in range(8):
        if cur == prev:
            break
        prev = cur
        cur = _VAR_RE.sub(repl, cur)
    return cur


def _from_images(text: str) -> list[str]:
    defaults = _arg_defaults(text)
    images: list[str] = []
    for match in _FROM_RE.finditer(text):
        raw = match.group(1)
        if raw.upper() == "scratch":
            continue
        # AS alias is separate token; FROM image AS name — group is first token only.
        expanded = _expand_vars(raw, defaults)
        # python:${PYTHON_VERSION}-slim-bookworm style after partial expand
        expanded = _expand_vars(expanded, defaults)
        images.append(expanded)
    return images


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_from_bases_pinned_by_sha256_digest(dockerfile: Path) -> None:
    """Given: Dockerfile under test. When: parse FROM lines. Then: each base ends with @sha256:<64 hex>."""
    assert dockerfile.is_file(), f"missing {dockerfile}"
    text = dockerfile.read_text(encoding="utf-8")
    images = _from_images(text)
    assert images, f"{dockerfile.name}: no FROM lines found"
    unpinned = [img for img in images if not _SHA256_SUFFIX.search(img)]
    assert not unpinned, (
        f"{dockerfile.name}: unpinned base image(s) (require @sha256:<64-hex>): {unpinned}"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_no_editable_pip_install_in_image_path(dockerfile: Path) -> None:
    """Given: image build instructions. When: scan RUN pip. Then: no -e/--editable installs."""
    text = dockerfile.read_text(encoding="utf-8")
    # Join line continuations for multi-line RUN pip install ...
    joined = text.replace("\\\n", " ")
    match = _EDITABLE_PIP.search(joined)
    assert match is None, (
        f"{dockerfile.name}: editable install forbidden in image path: {match.group(0)!r}"
    )


def test_requirements_lock_exists_with_sha256_hashes() -> None:
    """Given: hermetic dep policy. When: read requirements.lock. Then: file has --hash=sha256: pins."""
    assert LOCKFILE.is_file(), "requirements.lock missing (hash-locked lockfile required)"
    body = LOCKFILE.read_text(encoding="utf-8")
    assert "--hash=sha256:" in body, "requirements.lock must contain --hash=sha256: entries"
    # At least one concrete package pin (name==version).
    assert re.search(r"^[a-zA-Z0-9_.-]+==\S+", body, re.MULTILINE), (
        "requirements.lock must pin packages with =="
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_dockerfile_installs_via_hashed_lockfile(dockerfile: Path) -> None:
    """Given: Dockerfile install path. When: scan pip install. Then: uses lockfile + require-hashes."""
    text = dockerfile.read_text(encoding="utf-8")
    joined = text.replace("\\\n", " ")
    assert "requirements.lock" in text, (
        f"{dockerfile.name}: must COPY/install from requirements.lock"
    )
    assert _HASH_FLAG.search(joined) or "--require-hashes" in joined, (
        f"{dockerfile.name}: pip install must use --require-hashes (or lock hashes)"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_source_date_epoch_set(dockerfile: Path) -> None:
    """Given: reproducible build policy. When: scan ENV/ARG. Then: SOURCE_DATE_EPOCH is set."""
    text = dockerfile.read_text(encoding="utf-8")
    assert re.search(r"\bSOURCE_DATE_EPOCH\b", text), (
        f"{dockerfile.name}: SOURCE_DATE_EPOCH must be set for reproducible builds"
    )
