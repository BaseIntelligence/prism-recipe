"""Runtime self-measurement of the sealed surface (mech 5)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from prism_recipe.sealed_surface import (
    SealedSurfaceError,
    build_manifest,
    load_baked_manifest,
    resolve_repo_root,
    sealed_relative_paths,
    sha256_file,
)
from prism_recipe.sidecar.errors import SidecarError


def measure_sealed_manifest_hashes(*, root: Path | None = None) -> dict[str, str]:
    """Hash every sealed relative path on disk at answer time.

    Returns path → lowercase SHA-256 hex. Fail-closed if any sealed file is
    missing. This is the live measurement bound into the signed payload.
    """
    try:
        base = resolve_repo_root(root)
        files: dict[str, str] = {}
        missing: list[str] = []
        for rel in sealed_relative_paths():
            path = base / rel
            if not path.is_file():
                missing.append(rel)
                continue
            files[rel] = sha256_file(path)
        if missing:
            raise SidecarError(
                "sealed-surface files missing at measure time: " + ", ".join(missing)
            )
        return files
    except SealedSurfaceError as exc:
        raise SidecarError(str(exc)) from exc


def load_baked_hashes(*, root: Path | None = None) -> dict[str, str] | None:
    """Return baked manifest file hashes if present; else None (dev without bake)."""
    try:
        baked = load_baked_manifest(root)
    except SealedSurfaceError as exc:
        raise SidecarError(str(exc)) from exc
    if baked is None:
        return None
    files = baked.get("files")
    if not isinstance(files, dict):
        raise SidecarError("baked sealed manifest missing files map")
    return {str(k): str(v).lower() for k, v in files.items()}


def compare_live_to_baked(
    live: Mapping[str, str],
    baked: Mapping[str, str] | None,
) -> tuple[bool, tuple[str, ...]]:
    """Compare live measurement to baked hashes.

    Returns ``(match, mismatched_paths)``. When no baked manifest exists,
    match is True and mismatched is empty (nothing to diverge from yet).
    """
    if baked is None:
        return True, ()
    mismatched: list[str] = []
    for path, expected in baked.items():
        actual = live.get(path)
        if actual is None or actual.lower() != expected.lower():
            mismatched.append(path)
    for path in live:
        if path not in baked:
            mismatched.append(path)
    # Stable unique order
    ordered = tuple(dict.fromkeys(mismatched))
    return (len(ordered) == 0, ordered)


def measure_and_compare(
    *, root: Path | None = None
) -> tuple[dict[str, str], bool, tuple[str, ...]]:
    """Measure live hashes and compare to baked manifest in one step."""
    live = measure_sealed_manifest_hashes(root=root)
    baked = load_baked_hashes(root=root)
    match, mismatched = compare_live_to_baked(live, baked)
    return live, match, mismatched


def build_live_manifest_dict(*, root: Path | None = None) -> dict[str, object]:
    """Convenience: full sealed_surface.build_manifest for diagnostics."""
    try:
        return build_manifest(root)
    except SealedSurfaceError as exc:
        raise SidecarError(str(exc)) from exc
