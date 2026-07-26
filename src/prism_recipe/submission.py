"""Miner submission contract: git repo + pinned commit/tree SHA (BASE builds).

Schema: ``{repo_url, commit_sha, tree_sha, variant}`` with ``variant ∈ {cpu, cuda}``.

Contract validation only — no git clone. Full 40-char hex SHAs required (reject
branches/tags/short SHAs). ``tree_sha`` pins content against force-push rewrite.
HTTPS remotes only. Size ceiling (default 100 MiB), submodules, and Git-LFS are
enforced when a caller supplies ``SubmissionProbe``; network size probing is
deferred to the build/intake layer. Not a TEE or immutability claim.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, cast
from urllib.parse import urlparse

Variant = Literal["cpu", "cuda"]
VALID_VARIANTS: Final[frozenset[str]] = frozenset({"cpu", "cuda"})
DEFAULT_REPO_SIZE_CEILING_BYTES: Final[int] = 100 * 1024 * 1024

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_REF_PREFIX_RE = re.compile(r"^(refs/|heads/|tags/|origin/)", re.IGNORECASE)


class SubmissionError(ValueError):
    """Base class for miner submission contract violations."""

    def __init__(self, message: str, *, field: str, reason_code: str) -> None:
        super().__init__(message)
        self.field = field
        self.reason_code = reason_code


class BranchNameError(SubmissionError):
    """Branch/tag name supplied where a full commit SHA is required."""


class ShortShaError(SubmissionError):
    """Abbreviated hex SHA instead of a full 40-char SHA."""


class SshRemoteError(SubmissionError):
    """SSH remote (``git@`` / ``ssh://``) instead of HTTPS."""


class SubmoduleRejectedError(SubmissionError):
    """Probe reported Git submodules."""


class LfsRejectedError(SubmissionError):
    """Probe reported Git LFS objects."""


class InvalidShaError(SubmissionError):
    """SHA field present but not valid hex."""


class InvalidVariantError(SubmissionError):
    """``variant`` is not ``cpu`` or ``cuda``."""


class RepoSizeError(SubmissionError):
    """Probe-reported repo size exceeds the ceiling."""


class InvalidRepoUrlError(SubmissionError):
    """Non-HTTPS or otherwise unusable repository URL."""


@dataclass(frozen=True, slots=True)
class SubmissionProbe:
    """Deferred detection results (size / submodules / LFS). No clone here."""

    size_bytes: int | None = None
    has_submodules: bool | None = None
    has_lfs: bool | None = None


@dataclass(frozen=True, slots=True)
class MinerSubmission:
    """Normalized miner code pin for a BASE-owned image build."""

    repo_url: str
    commit_sha: str
    tree_sha: str
    variant: Variant

    def as_dict(self) -> dict[str, str]:
        return {
            "repo_url": self.repo_url,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "variant": self.variant,
        }


def _require_str(raw: Mapping[str, Any], field: str) -> str:
    if field not in raw or raw[field] is None:
        raise SubmissionError(f"{field} is required", field=field, reason_code="missing_field")
    value = raw[field]
    if not isinstance(value, str):
        raise SubmissionError(
            f"{field} must be a string, got {type(value).__name__}",
            field=field,
            reason_code="invalid_type",
        )
    stripped = value.strip()
    if not stripped:
        raise SubmissionError(
            f"{field} must be a non-empty string",
            field=field,
            reason_code="empty_field",
        )
    return stripped


def _classify_non_full_sha(value: str, *, field: str) -> SubmissionError:
    """Map an invalid pin to the most specific error type."""
    lowered = value.lower()
    if _REF_PREFIX_RE.match(value) or "/" in value:
        return BranchNameError(
            f"{field} must be a full 40-char commit SHA, not a branch/tag ref (got {value!r})",
            field=field,
            reason_code="branch_name",
        )
    if _HEX_RE.fullmatch(lowered):
        if len(lowered) < 40:
            return ShortShaError(
                f"{field} must be a full 40-char hex SHA, not a short SHA "
                f"(got length {len(lowered)})",
                field=field,
                reason_code="short_sha",
            )
        return InvalidShaError(
            f"{field} must be exactly 40 hex characters (got length {len(lowered)})",
            field=field,
            reason_code="invalid_sha",
        )
    # Length-40 non-hex looks like a corrupted SHA, not a branch name.
    if len(value) == 40:
        return InvalidShaError(
            f"{field} must be 40 hex characters (got non-hex value)",
            field=field,
            reason_code="invalid_sha",
        )
    return BranchNameError(
        f"{field} must be a full 40-char commit SHA, not a branch or tag name (got {value!r})",
        field=field,
        reason_code="branch_name",
    )


def _normalize_sha(value: str, *, field: str) -> str:
    lowered = value.strip().lower()
    if _FULL_SHA_RE.fullmatch(lowered):
        return lowered
    raise _classify_non_full_sha(value.strip(), field=field)


def _normalize_repo_url(value: str) -> str:
    raw = value.strip()
    if raw.startswith("git@"):
        raise SshRemoteError(
            f"repo_url must be an HTTPS remote; SSH remotes (git@…) are rejected (got {raw!r})",
            field="repo_url",
            reason_code="ssh_remote",
        )
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme == "ssh":
        raise SshRemoteError(
            f"repo_url must be an HTTPS remote; ssh:// remotes are rejected (got {raw!r})",
            field="repo_url",
            reason_code="ssh_remote",
        )
    if scheme in {"git", "http"}:
        raise InvalidRepoUrlError(
            f"repo_url must use HTTPS ({scheme}:// is rejected) (got {raw!r})",
            field="repo_url",
            reason_code="insecure_remote",
        )
    if scheme != "https":
        raise InvalidRepoUrlError(
            f"repo_url must be an HTTPS URL (got scheme={scheme!r}, value={raw!r})",
            field="repo_url",
            reason_code="invalid_remote",
        )
    if not parsed.netloc:
        raise InvalidRepoUrlError(
            f"repo_url is missing a host (got {raw!r})",
            field="repo_url",
            reason_code="invalid_remote",
        )
    path = parsed.path.rstrip("/")
    if not path:
        raise InvalidRepoUrlError(
            f"repo_url must include a repository path (got {raw!r})",
            field="repo_url",
            reason_code="invalid_remote",
        )
    return f"https://{parsed.netloc}{path}"


def _normalize_variant(value: str) -> Variant:
    lowered = value.strip().lower()
    if lowered not in VALID_VARIANTS:
        raise InvalidVariantError(
            f"variant must be one of {sorted(VALID_VARIANTS)}, got {value!r}",
            field="variant",
            reason_code="invalid_variant",
        )
    return cast(Variant, lowered)


def _apply_probe(probe: SubmissionProbe | None, *, size_ceiling_bytes: int) -> None:
    if probe is None:
        return
    if probe.has_submodules is True:
        raise SubmoduleRejectedError(
            "submodules are rejected for miner submissions (detected via submission probe)",
            field="repo_url",
            reason_code="submodule",
        )
    if probe.has_lfs is True:
        raise LfsRejectedError(
            "Git LFS objects are rejected for miner submissions (detected via submission probe)",
            field="repo_url",
            reason_code="git_lfs",
        )
    if probe.size_bytes is None:
        return
    if probe.size_bytes < 0:
        raise RepoSizeError(
            f"probe size_bytes must be >= 0, got {probe.size_bytes}",
            field="repo_url",
            reason_code="repo_size",
        )
    if probe.size_bytes > size_ceiling_bytes:
        raise RepoSizeError(
            f"repository size {probe.size_bytes} bytes exceeds ceiling {size_ceiling_bytes} bytes",
            field="repo_url",
            reason_code="repo_size",
        )


def validate_submission(
    raw: Mapping[str, Any],
    *,
    probe: SubmissionProbe | None = None,
    size_ceiling_bytes: int = DEFAULT_REPO_SIZE_CEILING_BYTES,
) -> MinerSubmission:
    """Parse and validate a miner submission mapping into a normalized record."""
    if size_ceiling_bytes < 1:
        raise ValueError("size_ceiling_bytes must be >= 1")
    repo_url = _normalize_repo_url(_require_str(raw, "repo_url"))
    commit_sha = _normalize_sha(_require_str(raw, "commit_sha"), field="commit_sha")
    tree_sha = _normalize_sha(_require_str(raw, "tree_sha"), field="tree_sha")
    variant = _normalize_variant(_require_str(raw, "variant"))
    _apply_probe(probe, size_ceiling_bytes=size_ceiling_bytes)
    return MinerSubmission(
        repo_url=repo_url,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        variant=variant,
    )


__all__ = [
    "DEFAULT_REPO_SIZE_CEILING_BYTES",
    "BranchNameError",
    "InvalidRepoUrlError",
    "InvalidShaError",
    "InvalidVariantError",
    "LfsRejectedError",
    "MinerSubmission",
    "RepoSizeError",
    "ShortShaError",
    "SshRemoteError",
    "SubmissionError",
    "SubmissionProbe",
    "SubmoduleRejectedError",
    "VALID_VARIANTS",
    "Variant",
    "validate_submission",
]
