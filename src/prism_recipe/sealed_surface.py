"""Sealed-surface manifest: build-time SHA-256 of harness / rules / data-window.

Mechanism 5 of prism-lium-image-attestation: the image bakes a manifest of every
sealed file hash. Runtime train paths must be a subset of that sealed set.

**M19 fix:** historically ``read_sealed_sources()`` sealed ``smoke_train.py``
while the CUDA path executed ``gpu_train.py``. This module seals both train
modules and fails closed when an executed path is not in the sealed set.

This is tamper-*evidence* for the sidecar / constation path — not
tamper-prevention. Do not claim immutability.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

Variant = Literal["cpu", "cuda"]

MANIFEST_SCHEMA_VERSION: Final[str] = "prism_sealed_surface.v1"
BAKED_MANIFEST_ENV: Final[str] = "PRISM_RECIPE_SEALED_MANIFEST"
DEFAULT_BAKED_MANIFEST_NAME: Final[str] = "sealed_surface_manifest.json"
HOME_ENV: Final[str] = "PRISM_RECIPE_HOME"

# Repo-relative POSIX paths that form the sealed surface (harness + rules +
# data-window pins + both train modules that can actually execute).
_SEALED_RELATIVE_PATHS: Final[tuple[str, ...]] = (
    ".rules/acceptance.md",
    ".rules/anti-cheat.md",
    ".rules/security.md",
    ".rules/training.md",
    "src/prism_recipe/arch/__init__.py",
    "src/prism_recipe/arch/tiny_1m.py",
    "src/prism_recipe/cli.py",
    "src/prism_recipe/config.py",  # egalitarian data-window pins
    "src/prism_recipe/gpu_train.py",  # CUDA executed train module (M19)
    "src/prism_recipe/harness.py",
    "src/prism_recipe/llm_gate.py",
    "src/prism_recipe/loader.py",
    "src/prism_recipe/sealed_surface.py",
    "src/prism_recipe/smoke_train.py",  # CPU smoke executed train module
)

# Modules that actually run on the train path for each image variant.
_VARIANT_EXECUTED_TRAIN: Final[Mapping[Variant, tuple[str, ...]]] = {
    "cpu": ("src/prism_recipe/smoke_train.py",),
    "cuda": ("src/prism_recipe/gpu_train.py",),
}

_VARIANT_EXECUTED_SHARED: Final[tuple[str, ...]] = (
    "src/prism_recipe/arch/tiny_1m.py",
    "src/prism_recipe/config.py",
    "src/prism_recipe/harness.py",
    "src/prism_recipe/llm_gate.py",
    "src/prism_recipe/loader.py",
    "src/prism_recipe/sealed_surface.py",
)


class SealedSurfaceError(RuntimeError):
    """Fail-closed sealed-surface violation (unsealed exec or missing file)."""


def sealed_relative_paths() -> tuple[str, ...]:
    """Return the canonical sealed-surface path list (repo-relative POSIX)."""
    return _SEALED_RELATIVE_PATHS


def executed_paths_for_variant(variant: Variant | str) -> tuple[str, ...]:
    """Return the modules that actually execute for ``cpu`` or ``cuda`` train."""
    key = _normalize_variant(variant)
    train = _VARIANT_EXECUTED_TRAIN[key]
    # Stable order: shared then train-specific.
    return tuple(dict.fromkeys((*_VARIANT_EXECUTED_SHARED, *train)))


def _normalize_variant(variant: Variant | str) -> Variant:
    raw = str(variant).strip().lower()
    if raw in {"cpu", "cpu-smoke", "smoke"}:
        return "cpu"
    if raw in {"cuda", "gpu", "gpu-short"}:
        return "cuda"
    raise SealedSurfaceError(f"unknown sealed-surface variant: {variant!r}")


def resolve_repo_root(root: Path | None = None) -> Path:
    """Resolve checkout / image app root containing ``src/prism_recipe`` + ``.rules``."""
    if root is not None:
        return root.resolve()
    env_home = os.environ.get(HOME_ENV, "").strip()
    if env_home:
        candidate = Path(env_home)
        if (candidate / "src" / "prism_recipe").is_dir():
            return candidate.resolve()
    # src/prism_recipe/sealed_surface.py → parents[2] == repo root in checkout
    # and /app in the image (WORKDIR /app, COPY src ./src).
    here = Path(__file__).resolve()
    package_parent = here.parents[1]  # .../src/prism_recipe or site-packages/prism_recipe
    # Prefer tree layout: <root>/src/prism_recipe/this.py
    if package_parent.name == "prism_recipe" and package_parent.parent.name == "src":
        return package_parent.parent.parent
    # Editable / wheel install: fall back to PRISM_RECIPE_HOME or cwd.
    cwd = Path.cwd()
    if (cwd / "src" / "prism_recipe").is_dir():
        return cwd.resolve()
    raise SealedSurfaceError(
        "cannot resolve recipe repo root (set PRISM_RECIPE_HOME or run from checkout)"
    )


def sha256_file(path: Path) -> str:
    """Return lowercase hex SHA-256 of file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path | None = None) -> dict[str, Any]:
    """Build a sealed-surface manifest dict from files under ``root``."""
    base = resolve_repo_root(root)
    files: dict[str, str] = {}
    missing: list[str] = []
    for rel in _SEALED_RELATIVE_PATHS:
        path = base / rel
        if not path.is_file():
            missing.append(rel)
            continue
        files[rel] = sha256_file(path)
    if missing:
        raise SealedSurfaceError(
            "sealed-surface files missing at build/runtime: " + ", ".join(missing)
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": files,
        "variants": {
            "cpu": {
                "executed": list(executed_paths_for_variant("cpu")),
            },
            "cuda": {
                "executed": list(executed_paths_for_variant("cuda")),
            },
        },
    }


def bake_manifest(root: Path | None, out_path: Path) -> dict[str, Any]:
    """Write manifest JSON to ``out_path`` (build-time bake). Returns the dict."""
    manifest = build_manifest(root)
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic JSON: sorted keys, stable separators, trailing newline.
    payload = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    return manifest


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load a previously baked sealed-surface manifest JSON."""
    p = Path(path)
    if not p.is_file():
        raise SealedSurfaceError(f"sealed-surface manifest not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SealedSurfaceError(f"sealed-surface manifest invalid JSON: {p}") from exc
    if not isinstance(data, dict):
        raise SealedSurfaceError("sealed-surface manifest root must be an object")
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise SealedSurfaceError("sealed-surface manifest missing files map")
    return data


def default_baked_manifest_path(root: Path | None = None) -> Path:
    """Preferred on-disk location of the baked manifest (image or checkout)."""
    env = os.environ.get(BAKED_MANIFEST_ENV, "").strip()
    if env:
        return Path(env)
    base = resolve_repo_root(root)
    return base / DEFAULT_BAKED_MANIFEST_NAME


def load_baked_manifest(root: Path | None = None) -> dict[str, Any] | None:
    """Load baked manifest if present; return None when not yet baked (dev)."""
    path = default_baked_manifest_path(root)
    if not path.is_file():
        return None
    return load_manifest(path)


def sealed_set(
    manifest: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> frozenset[str]:
    """Return the set of sealed relative paths from manifest or canonical list."""
    if manifest is None:
        baked = load_baked_manifest(root)
        if baked is not None:
            manifest = baked
    if manifest is not None:
        files = manifest.get("files")
        if isinstance(files, dict):
            return frozenset(str(k) for k in files)
    return frozenset(_SEALED_RELATIVE_PATHS)


def assert_executed_module_sealed(
    relative_path: str,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> None:
    """Fail closed when ``relative_path`` is not in the sealed set."""
    rel = relative_path.replace("\\", "/").lstrip("./")
    sealed = sealed_set(manifest, root=root)
    if rel not in sealed:
        raise SealedSurfaceError(
            f"executed path not in sealed set (fail closed): {rel} "
            f"(unsealed execution rejected; sealed has {len(sealed)} paths)"
        )


def assert_variant_execution_sealed(
    variant: Variant | str,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> None:
    """Fail closed unless every executed path for ``variant`` is sealed."""
    if manifest is None:
        baked = load_baked_manifest(root)
        manifest = baked if baked is not None else build_manifest(root)
    sealed = sealed_set(manifest, root=root)
    executed = executed_paths_for_variant(variant)
    missing = [p for p in executed if p not in sealed]
    if missing:
        raise SealedSurfaceError(
            f"executed path(s) not in sealed set for variant={variant!r}: "
            + ", ".join(missing)
        )
    # When a baked manifest is present, also verify on-disk hashes match.
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if isinstance(files, dict) and files:
        base = resolve_repo_root(root)
        for rel in executed:
            digest = files.get(rel)
            if not isinstance(digest, str):
                raise SealedSurfaceError(f"sealed manifest missing digest for {rel}")
            path = base / rel
            if not path.is_file():
                raise SealedSurfaceError(f"sealed executed file missing on disk: {rel}")
            actual = sha256_file(path)
            if actual != digest:
                raise SealedSurfaceError(
                    f"sealed file hash mismatch for {rel}: "
                    f"manifest={digest[:12]}… actual={actual[:12]}…"
                )


def train_module_relative_path(variant: Variant | str) -> str:
    """Relative path of the primary train module for the variant."""
    key = _normalize_variant(variant)
    return _VARIANT_EXECUTED_TRAIN[key][0]


def read_sealed_sources(
    *,
    variant: Variant | str = "cpu",
    root: Path | None = None,
) -> tuple[str, str]:
    """Load sealed architecture + train source text for the LLM gate.

    ``variant`` selects which train module is sealed into the gate prompt:
    - ``cpu`` → ``smoke_train.py``
    - ``cuda`` → ``gpu_train.py`` (M19: must match the module that actually runs)
    """
    base = resolve_repo_root(root)
    arch_rel = "src/prism_recipe/arch/tiny_1m.py"
    train_rel = train_module_relative_path(variant)
    # Fail closed if either path is outside the sealed set.
    assert_executed_module_sealed(arch_rel, root=base)
    assert_executed_module_sealed(train_rel, root=base)
    arch_path = base / arch_rel
    train_path = base / train_rel
    if not arch_path.is_file() or not train_path.is_file():
        raise SealedSurfaceError(
            f"sealed sources missing: arch={arch_path} train={train_path}"
        )
    return (
        arch_path.read_text(encoding="utf-8"),
        train_path.read_text(encoding="utf-8"),
    )


def detect_runtime_variant() -> Variant:
    """Infer cpu vs cuda from env flags used by harness / CLI."""
    gpu = os.environ.get("PRISM_RECIPE_GPU_SHORT", "").strip().lower()
    if gpu in {"1", "true", "yes", "on"}:
        return "cuda"
    explicit = os.environ.get("PRISM_RECIPE_VARIANT", "").strip().lower()
    if explicit:
        return _normalize_variant(explicit)
    return "cpu"


def ensure_execution_sealed(
    *,
    variant: Variant | str | None = None,
    root: Path | None = None,
) -> Variant:
    """Resolve variant and fail closed if its exec paths are not sealed.

    Call at harness train entry before starting smoke or gpu train.
    """
    resolved = detect_runtime_variant() if variant is None else _normalize_variant(variant)
    assert_variant_execution_sealed(resolved, root=root)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m prism_recipe.sealed_surface bake|show|verify``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        sys.stdout.write(
            "usage: python -m prism_recipe.sealed_surface "
            "{bake [--out PATH]|show|verify} [--root PATH]\n"
        )
        return 0

    root: Path | None = None
    if "--root" in args:
        idx = args.index("--root")
        root = Path(args[idx + 1])
        del args[idx : idx + 2]

    cmd = args[0]
    if cmd == "bake":
        out = default_baked_manifest_path(root)
        if "--out" in args:
            idx = args.index("--out")
            out = Path(args[idx + 1])
        manifest = bake_manifest(root, out)
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "path": str(out),
                    "file_count": len(manifest["files"]),
                    "variants": manifest["variants"],
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    if cmd == "show":
        manifest = load_baked_manifest(root) or build_manifest(root)
        sys.stdout.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return 0

    if cmd == "verify":
        manifest = load_baked_manifest(root) or build_manifest(root)
        for variant in ("cpu", "cuda"):
            assert_variant_execution_sealed(variant, manifest=manifest, root=root)
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "cpu_executed": list(executed_paths_for_variant("cpu")),
                    "cuda_executed": list(executed_paths_for_variant("cuda")),
                    "sealed_count": len(sealed_set(manifest)),
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    sys.stderr.write(f"unknown command: {cmd}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BAKED_MANIFEST_ENV",
    "DEFAULT_BAKED_MANIFEST_NAME",
    "HOME_ENV",
    "MANIFEST_SCHEMA_VERSION",
    "SealedSurfaceError",
    "Variant",
    "assert_executed_module_sealed",
    "assert_variant_execution_sealed",
    "bake_manifest",
    "build_manifest",
    "default_baked_manifest_path",
    "detect_runtime_variant",
    "ensure_execution_sealed",
    "executed_paths_for_variant",
    "load_baked_manifest",
    "load_manifest",
    "main",
    "read_sealed_sources",
    "resolve_repo_root",
    "sealed_relative_paths",
    "sealed_set",
    "sha256_file",
    "train_module_relative_path",
]
