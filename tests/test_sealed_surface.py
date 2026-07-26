"""Sealed-surface manifest + seal/exec mismatch (M19) — TDD contract.

Scenarios
---------
S1 happy: executed-path ⊆ sealed-set for cpu and cuda variants.
S2 happy: build-time SHA-256 manifest covers harness / rules / data-window pins.
S3 edge: unsealed executed path fails closed (SealedSurfaceError).
S4 regression: CUDA gate sources use gpu_train.py (not smoke_train.py only).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_executed_path_subset_of_sealed_set_for_cpu_and_cuda() -> None:
    """S1: Given sealed surface, When resolve exec paths per variant, Then ⊆ sealed-set."""
    from prism_recipe.sealed_surface import (
        executed_paths_for_variant,
        sealed_relative_paths,
    )

    sealed = set(sealed_relative_paths())
    for variant in ("cpu", "cuda"):
        executed = set(executed_paths_for_variant(variant))
        assert executed, f"{variant}: executed set must be non-empty"
        missing = executed - sealed
        assert not missing, (
            f"{variant}: executed path(s) not in sealed set (M19): {sorted(missing)}"
        )


def test_manifest_sha256_covers_harness_rules_and_data_window() -> None:
    """S2: Given repo root, When bake manifest, Then every sealed file has sha256 hex."""
    from prism_recipe.sealed_surface import (
        MANIFEST_SCHEMA_VERSION,
        build_manifest,
        sealed_relative_paths,
    )

    manifest = build_manifest(REPO_ROOT)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    files = manifest["files"]
    assert isinstance(files, dict)
    expected = set(sealed_relative_paths())
    assert set(files) == expected

    # Rules surface
    for name in (
        ".rules/training.md",
        ".rules/anti-cheat.md",
        ".rules/security.md",
        ".rules/acceptance.md",
    ):
        assert name in files

    # Harness + train modules (both variants)
    for name in (
        "src/prism_recipe/harness.py",
        "src/prism_recipe/smoke_train.py",
        "src/prism_recipe/gpu_train.py",
        "src/prism_recipe/loader.py",
        "src/prism_recipe/llm_gate.py",
        "src/prism_recipe/config.py",  # data-window pins
        "src/prism_recipe/arch/tiny_1m.py",
    ):
        assert name in files, f"missing sealed path: {name}"

    for rel, digest in files.items():
        assert isinstance(digest, str) and len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
        raw = (REPO_ROOT / rel).read_bytes()
        assert digest == hashlib.sha256(raw).hexdigest(), f"hash mismatch for {rel}"


def test_unsealed_executed_path_fails_closed() -> None:
    """S3: Given baked manifest, When assert unsealed module executes, Then SealedSurfaceError."""
    from prism_recipe.sealed_surface import (
        SealedSurfaceError,
        assert_executed_module_sealed,
        build_manifest,
    )

    manifest = build_manifest(REPO_ROOT)
    with pytest.raises(SealedSurfaceError, match="unsealed|not in sealed"):
        assert_executed_module_sealed(
            "src/prism_recipe/not_a_real_train.py",
            manifest=manifest,
        )


def test_cuda_read_sealed_sources_includes_gpu_train_not_only_smoke() -> None:
    """S4: Given cuda variant, When read_sealed_sources, Then training source is gpu_train."""
    from prism_recipe.sealed_surface import read_sealed_sources

    arch_src, train_src = read_sealed_sources(variant="cuda", root=REPO_ROOT)
    assert "build_tiny_1m" in arch_src or "transformer-tiny-1m" in arch_src or "Tiny" in arch_src
    assert "run_gpu_short_train" in train_src
    assert "def run_smoke_train" not in train_src  # must not be smoke-only seal

    arch_cpu, train_cpu = read_sealed_sources(variant="cpu", root=REPO_ROOT)
    assert "def run_smoke_train" in train_cpu
    assert arch_cpu == arch_src  # same sealed arch baseline


def test_assert_variant_exec_paths_ok_for_both() -> None:
    """S1 surface: assert_variant_execution_sealed passes for cpu and cuda on real tree."""
    from prism_recipe.sealed_surface import (
        assert_variant_execution_sealed,
        build_manifest,
    )

    manifest = build_manifest(REPO_ROOT)
    assert_variant_execution_sealed("cpu", manifest=manifest)
    assert_variant_execution_sealed("cuda", manifest=manifest)


def test_bake_manifest_roundtrip_json(tmp_path: Path) -> None:
    """S2 edge: bake to JSON and reload yields identical file digests."""
    from prism_recipe.sealed_surface import bake_manifest, load_manifest

    out = tmp_path / "sealed_surface_manifest.json"
    baked = bake_manifest(REPO_ROOT, out)
    loaded = load_manifest(out)
    assert loaded["files"] == baked["files"]
    assert out.is_file()
    # JSON is stable enough to parse
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["files"]["src/prism_recipe/gpu_train.py"]
