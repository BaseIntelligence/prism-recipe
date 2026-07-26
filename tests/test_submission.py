"""Miner submission contract unit tests (shape validation only, no git clone)."""

from __future__ import annotations

import pytest

from prism_recipe.submission import (
    DEFAULT_REPO_SIZE_CEILING_BYTES,
    BranchNameError,
    InvalidShaError,
    InvalidVariantError,
    LfsRejectedError,
    MinerSubmission,
    RepoSizeError,
    ShortShaError,
    SshRemoteError,
    SubmissionError,
    SubmissionProbe,
    SubmoduleRejectedError,
    validate_submission,
)

# Fixed full SHAs for fixtures (40-char lowercase hex).
COMMIT_SHA = "a" * 40
TREE_SHA = "b" * 40
HTTPS_REPO = "https://github.com/example/miner-arch.git"


def _valid(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "repo_url": HTTPS_REPO,
        "commit_sha": COMMIT_SHA,
        "tree_sha": TREE_SHA,
        "variant": "cpu",
    }
    base.update(overrides)
    return base


def test_accepts_valid_https_cpu_submission() -> None:
    """Given valid HTTPS + full SHAs + cpu, When validate, Then normalized MinerSubmission."""
    result = validate_submission(_valid())
    assert isinstance(result, MinerSubmission)
    assert result.repo_url == "https://github.com/example/miner-arch.git"
    assert result.commit_sha == COMMIT_SHA
    assert result.tree_sha == TREE_SHA
    assert result.variant == "cpu"
    assert result.as_dict() == {
        "repo_url": "https://github.com/example/miner-arch.git",
        "commit_sha": COMMIT_SHA,
        "tree_sha": TREE_SHA,
        "variant": "cpu",
    }


def test_accepts_cuda_variant_and_normalizes_sha_case() -> None:
    upper_commit = "C" * 40
    upper_tree = "D" * 40
    result = validate_submission(
        _valid(
            commit_sha=upper_commit,
            tree_sha=upper_tree,
            variant="cuda",
            repo_url="https://github.com/example/miner-arch",
        )
    )
    assert result.variant == "cuda"
    assert result.commit_sha == "c" * 40
    assert result.tree_sha == "d" * 40
    assert result.repo_url.endswith(".git") is False or result.repo_url.startswith("https://")


def test_rejects_branch_name_as_commit_sha_distinctly() -> None:
    """Given commit_sha='main', When validate, Then BranchNameError naming commit_sha."""
    with pytest.raises(BranchNameError, match="commit_sha") as raised:
        validate_submission(_valid(commit_sha="main"))
    err = raised.value
    assert isinstance(err, SubmissionError)
    assert err.field == "commit_sha"
    assert "branch" in str(err).lower() or err.reason_code == "branch_name"


def test_rejects_refs_heads_as_branch_name() -> None:
    with pytest.raises(BranchNameError, match="commit_sha"):
        validate_submission(_valid(commit_sha="refs/heads/feature-x"))


def test_rejects_short_sha_distinctly() -> None:
    """Given 7-char hex commit_sha, When validate, Then ShortShaError."""
    with pytest.raises(ShortShaError, match="commit_sha") as raised:
        validate_submission(_valid(commit_sha="abc1234"))
    err = raised.value
    assert err.field == "commit_sha"
    assert err.reason_code == "short_sha"
    assert "40" in str(err)


def test_rejects_short_tree_sha_distinctly() -> None:
    with pytest.raises(ShortShaError, match="tree_sha") as raised:
        validate_submission(_valid(tree_sha="deadbee"))
    assert raised.value.field == "tree_sha"
    assert raised.value.reason_code == "short_sha"


def test_rejects_missing_tree_sha() -> None:
    raw = _valid()
    del raw["tree_sha"]
    with pytest.raises(SubmissionError, match="tree_sha"):
        validate_submission(raw)


def test_rejects_non_hex_sha() -> None:
    with pytest.raises(InvalidShaError, match="commit_sha"):
        validate_submission(_valid(commit_sha="z" * 40))


def test_rejects_ssh_git_at_remote_distinctly() -> None:
    """Given git@ SSH remote, When validate, Then SshRemoteError."""
    with pytest.raises(SshRemoteError, match="repo_url") as raised:
        validate_submission(_valid(repo_url="git@github.com:example/miner-arch.git"))
    err = raised.value
    assert err.field == "repo_url"
    assert err.reason_code == "ssh_remote"
    assert "https" in str(err).lower()


def test_rejects_ssh_scheme_remote() -> None:
    with pytest.raises(SshRemoteError, match="repo_url"):
        validate_submission(_valid(repo_url="ssh://git@github.com/example/miner-arch.git"))


def test_rejects_http_and_git_scheme_remotes() -> None:
    with pytest.raises(SubmissionError, match="repo_url"):
        validate_submission(_valid(repo_url="http://github.com/example/miner-arch.git"))
    with pytest.raises(SubmissionError, match="repo_url"):
        validate_submission(_valid(repo_url="git://github.com/example/miner-arch.git"))


def test_rejects_submodule_when_probe_detects() -> None:
    """Given probe.has_submodules=True, When validate, Then SubmoduleRejectedError."""
    with pytest.raises(SubmoduleRejectedError) as raised:
        validate_submission(
            _valid(),
            probe=SubmissionProbe(has_submodules=True),
        )
    err = raised.value
    assert err.reason_code == "submodule"
    assert "submodule" in str(err).lower()


def test_rejects_git_lfs_when_probe_detects() -> None:
    with pytest.raises(LfsRejectedError) as raised:
        validate_submission(_valid(), probe=SubmissionProbe(has_lfs=True))
    assert raised.value.reason_code == "git_lfs"


def test_rejects_repo_over_size_ceiling_when_probe_provides_size() -> None:
    too_big = DEFAULT_REPO_SIZE_CEILING_BYTES + 1
    with pytest.raises(RepoSizeError) as raised:
        validate_submission(
            _valid(),
            probe=SubmissionProbe(size_bytes=too_big),
        )
    assert raised.value.reason_code == "repo_size"
    assert str(DEFAULT_REPO_SIZE_CEILING_BYTES) in str(raised.value) or "100" in str(
        raised.value
    )


def test_accepts_when_probe_size_within_ceiling() -> None:
    result = validate_submission(
        _valid(),
        probe=SubmissionProbe(
            size_bytes=DEFAULT_REPO_SIZE_CEILING_BYTES,
            has_submodules=False,
            has_lfs=False,
        ),
    )
    assert result.commit_sha == COMMIT_SHA


def test_rejects_invalid_variant() -> None:
    with pytest.raises(InvalidVariantError, match="variant") as raised:
        validate_submission(_valid(variant="rocm"))
    assert raised.value.field == "variant"
    assert raised.value.reason_code == "invalid_variant"


def test_branch_short_ssh_submodule_errors_are_distinct_types() -> None:
    """Acceptance: branch, short SHA, SSH, submodule each raise distinctly."""
    cases: list[tuple[type[SubmissionError], dict[str, object], SubmissionProbe | None]] = [
        (BranchNameError, _valid(commit_sha="main"), None),
        (ShortShaError, _valid(commit_sha="abc1234"), None),
        (SshRemoteError, _valid(repo_url="git@github.com:x/y.git"), None),
        (SubmoduleRejectedError, _valid(), SubmissionProbe(has_submodules=True)),
    ]
    seen: set[type[BaseException]] = set()
    for exc_type, raw, probe in cases:
        with pytest.raises(exc_type):
            validate_submission(raw, probe=probe)
        seen.add(exc_type)
    assert len(seen) == 4


def test_size_probe_deferred_without_probe_metadata() -> None:
    """Without network probe, size/submodule/LFS checks are skipped (shape-only)."""
    # Must still accept — deferred size probe is documented on the module.
    result = validate_submission(_valid(variant="cuda"))
    assert result.variant == "cuda"
