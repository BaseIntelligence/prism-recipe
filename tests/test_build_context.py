"""Build-context seal constraints: miner confined; sealed surfaces non-overwritable.

TDD for checkbox 4 — collision rejected at build-prep with named path; seal hashes
intact after assemble; miner supplies no Dockerfile/base image; network policy
documented and validated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from prism_recipe.build_context import (
    ALLOWED_BUILD_REGISTRIES,
    MINER_SUBTREE,
    SEALED_RELATIVE_PATHS,
    BuildContextError,
    ForbiddenMinerArtifactError,
    SealCollisionError,
    assemble_build_context,
    sealed_file_sha256,
    validate_dockerfile_network_policy,
    validate_miner_tree,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, body: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _benign_miner(tmp: Path) -> Path:
    root = tmp / "miner_src"
    _write(root / "train.py", "def train():\n    return 1\n")
    _write(root / "README.md", "miner code\n")
    return root


# --- S1 happy: seal intact -------------------------------------------------


def test_assemble_benign_miner_keeps_sealed_hashes_intact(tmp_path: Path) -> None:
    """Given benign miner tree, When assemble, Then sealed file digests match recipe originals."""
    miner = _benign_miner(tmp_path)
    dest = tmp_path / "ctx"
    result = assemble_build_context(
        recipe_root=ROOT,
        miner_root=miner,
        dest=dest,
    )
    assert result == dest
    assert (dest / MINER_SUBTREE / "train.py").is_file()
    for rel in SEALED_RELATIVE_PATHS:
        sealed = dest / rel.rstrip("/")
        if rel.endswith("/"):
            assert sealed.is_dir(), f"missing sealed dir in context: {rel}"
            continue
        assert sealed.is_file(), f"missing sealed path in context: {rel}"
        original = ROOT / rel
        assert sealed_file_sha256(sealed) == sealed_file_sha256(original), (
            f"sealed surface mutated for {rel}"
        )


def test_miner_content_confined_to_miner_subtree(tmp_path: Path) -> None:
    """Given assemble, When list dest, Then miner files only under miner/ not at sealed roots."""
    miner = _benign_miner(tmp_path)
    dest = tmp_path / "ctx"
    assemble_build_context(recipe_root=ROOT, miner_root=miner, dest=dest)
    assert (dest / MINER_SUBTREE / "train.py").is_file()
    assert not (dest / "train.py").exists()
    # Must not land miner files on top of package root.
    assert not (dest / "src" / "train.py").exists()


# --- S2 collision rejected with named path ---------------------------------


def test_collision_harness_py_rejected_with_named_path(tmp_path: Path) -> None:
    """Given miner harness.py, When validate, Then SealCollisionError names path."""
    miner = tmp_path / "evil"
    _write(miner / "harness.py", "# overwrite attempt\n")
    with pytest.raises(SealCollisionError, match="harness") as raised:
        validate_miner_tree(miner)
    err = raised.value
    assert isinstance(err, BuildContextError)
    assert "harness" in err.path.lower() or "harness" in str(err).lower()
    assert err.path  # named path required




def test_collision_harness_dir_rejected(tmp_path: Path) -> None:
    """Given miner harness/ directory, When validate, Then SealCollisionError names it."""
    miner = tmp_path / "evil"
    _write(miner / "harness" / "plugin.py", "bad\n")
    with pytest.raises(SealCollisionError) as raised:
        validate_miner_tree(miner)
    assert "harness" in raised.value.path or "harness" in str(raised.value).lower()


def test_collision_nested_sealed_path_rejected(tmp_path: Path) -> None:
    """Given nested sealed harness path in miner, When validate, Then named collision."""
    miner = tmp_path / "evil"
    _write(miner / "src" / "prism_recipe" / "harness.py", "bad\n")
    with pytest.raises(SealCollisionError) as raised:
        validate_miner_tree(miner)
    assert "harness" in raised.value.path or "harness" in str(raised.value)


def test_collision_rules_dir_rejected(tmp_path: Path) -> None:
    miner = tmp_path / "evil"
    _write(miner / ".rules" / "training.md", "mutated\n")
    with pytest.raises(SealCollisionError) as raised:
        validate_miner_tree(miner)
    assert ".rules" in raised.value.path or "rules" in str(raised.value).lower()


def test_assemble_rejects_collision_before_copy(tmp_path: Path) -> None:
    miner = tmp_path / "evil"
    _write(miner / "loader.py", "bad\n")
    dest = tmp_path / "ctx"
    with pytest.raises(SealCollisionError, match="loader"):
        assemble_build_context(recipe_root=ROOT, miner_root=miner, dest=dest)
    assert not dest.exists() or not any(dest.rglob("loader.py"))


# --- S3 miner supplies no Dockerfile / base image ---------------------------


def test_miner_dockerfile_rejected(tmp_path: Path) -> None:
    """Given miner Dockerfile, When validate, Then ForbiddenMinerArtifactError."""
    miner = tmp_path / "evil"
    _write(miner / "train.py", "ok\n")
    _write(miner / "Dockerfile", "FROM evil\n")
    with pytest.raises(ForbiddenMinerArtifactError, match="Dockerfile") as raised:
        validate_miner_tree(miner)
    assert "Dockerfile" in raised.value.path


def test_miner_dockerfile_cuda_rejected(tmp_path: Path) -> None:
    miner = tmp_path / "evil"
    _write(miner / "Dockerfile.cuda", "FROM evil\n")
    with pytest.raises(ForbiddenMinerArtifactError, match="Dockerfile"):
        validate_miner_tree(miner)


def test_miner_base_image_pin_file_rejected(tmp_path: Path) -> None:
    """Miner must not supply a base-image override artifact."""
    miner = tmp_path / "evil"
    _write(miner / "BASE_IMAGE", "python:latest\n")
    with pytest.raises(ForbiddenMinerArtifactError) as raised:
        validate_miner_tree(miner)
    assert "BASE_IMAGE" in raised.value.path or "base" in str(raised.value).lower()


# --- S5 network policy ------------------------------------------------------


def test_allowed_build_registries_is_nonempty_pinned_set() -> None:
    assert isinstance(ALLOWED_BUILD_REGISTRIES, frozenset)
    assert ALLOWED_BUILD_REGISTRIES
    # docker hub canonical hosts used by current Dockerfiles
    assert (
        "docker.io" in ALLOWED_BUILD_REGISTRIES
        or "registry-1.docker.io" in ALLOWED_BUILD_REGISTRIES
    )


@pytest.mark.parametrize(
    "dockerfile",
    [ROOT / "Dockerfile", ROOT / "Dockerfile.cuda"],
    ids=lambda p: p.name,
)
def test_dockerfile_documents_and_satisfies_network_policy(dockerfile: Path) -> None:
    """Given hermetic Dockerfiles, When validate network policy, Then pass."""
    assert dockerfile.is_file()
    text = dockerfile.read_text(encoding="utf-8")
    validate_dockerfile_network_policy(text, path=dockerfile.name)
    # Policy comment must name deny-by-default / pinned registries.
    lowered = text.lower()
    assert "network" in lowered
    assert "registry" in lowered or "registries" in lowered


def test_network_policy_rejects_unpinned_from() -> None:
    bad = "# network denied except pinned registries\nFROM python:3.12-slim\n"
    with pytest.raises(BuildContextError, match="sha256|unpinned|registry"):
        validate_dockerfile_network_policy(bad, path="bad.Dockerfile")


def test_sealed_relative_paths_cover_harness_rules_loader_gate_arch_train() -> None:
    """Shared constant for todo 4 + todo 5: sealed surfaces named explicitly."""
    joined = "\n".join(sorted(SEALED_RELATIVE_PATHS))
    for needle in (
        "harness.py",
        "loader.py",
        "llm_gate.py",
        "tiny_1m.py",
        "smoke_train.py",
        "gpu_train.py",
        ".rules/",
    ):
        assert any(
            needle.rstrip("/") in p or p.startswith(needle) or needle in p
            for p in SEALED_RELATIVE_PATHS
        ), f"SEALED_RELATIVE_PATHS missing {needle}: {joined}"
    # Every listed file path must exist in the recipe tree (dirs end with /).
    for rel in SEALED_RELATIVE_PATHS:
        path = ROOT / rel.rstrip("/")
        assert path.exists(), f"sealed path missing from recipe: {rel}"


def test_dockerfile_copies_miner_before_sealed_src() -> None:
    """Given Dockerfile, When scan COPY order, Then miner subtree precedes sealed src/.rules."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    # Strip comments for ordering of real COPY instructions.
    lines = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if stripped.upper().startswith("COPY "):
            lines.append(stripped)
    joined = "\n".join(lines)
    miner_idx = next(i for i, ln in enumerate(lines) if re.search(r"\bminer\b", ln))
    src_idx = next(i for i, ln in enumerate(lines) if re.search(r"\bsrc\b", ln))
    rules_idx = next(i for i, ln in enumerate(lines) if ".rules" in ln)
    assert miner_idx < src_idx, f"miner must COPY before src:\n{joined}"
    assert miner_idx < rules_idx, f"miner must COPY before .rules:\n{joined}"
