"""Build-context assembly: confine miner code; protect sealed surfaces.

BASE owns the image build. Miner content is staged only under ``miner/`` (image
``/app/miner``). Sealed harness / rules / loader / gate / arch / train modules
are copied **after** miner content from the recipe tree and must never be
supplied or overwritten by the miner.

This module is build-prep only — no git clone. Collisions with sealed paths are
rejected with a named path before any copy into the staging directory.

Build network policy: deny by default except pinned registries listed in
``ALLOWED_BUILD_REGISTRIES`` (enforced here for Dockerfile text + documented in
Dockerfile comments). Tamper-evidence, not tamper-prevention.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Shared constants (todo 4 build-context + todo 5 seal manifest)
# ---------------------------------------------------------------------------

MINER_SUBTREE: Final[str] = "miner"

# Prefer the canonical sealed file list from sealed_surface (todo 5). Fall back
# to a local list if that module is unavailable during partial checkouts.
try:
    from prism_recipe.sealed_surface import sealed_relative_paths as _sealed_paths_fn

    _SEALED_FILES: Final[tuple[str, ...]] = _sealed_paths_fn()
except Exception:  # pragma: no cover - bootstrap / partial tree
    _SEALED_FILES = (
        "src/prism_recipe/harness.py",
        "src/prism_recipe/loader.py",
        "src/prism_recipe/llm_gate.py",
        "src/prism_recipe/smoke_train.py",
        "src/prism_recipe/gpu_train.py",
        "src/prism_recipe/arch/tiny_1m.py",
        "src/prism_recipe/arch/__init__.py",
        ".rules/training.md",
        ".rules/security.md",
        ".rules/acceptance.md",
        ".rules/anti-cheat.md",
    )

# Relative to recipe / assembled build context root. Trailing "/" = directory
# prefix (any file under it is sealed). File paths are exact sealed surfaces.
SEALED_RELATIVE_PATHS: Final[frozenset[str]] = frozenset(
    {
        *_SEALED_FILES,
        ".rules/",  # whole rules tree is sealed (collision prefix)
    }
)

# Path-component stems that must not appear in a miner tree (e.g. harness/).
SEALED_PATH_STEMS: Final[frozenset[str]] = frozenset(
    {
        "harness",
        "loader",
        "llm_gate",
        "smoke_train",
        "gpu_train",
        "tiny_1m",
        "sealed_surface",
        ".rules",
        "rules",
    }
)

# Hostnames miners/build may pull from. Everything else is denied.
ALLOWED_BUILD_REGISTRIES: Final[frozenset[str]] = frozenset(
    {
        "docker.io",
        "registry-1.docker.io",
        "ghcr.io",
        "nvidia.com",  # future CUDA mirrors if digest-pinned
        "nvcr.io",
    }
)

# Filenames the miner must never ship (BASE owns image definition).
_FORBIDDEN_MINER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "dockerfile",
        "dockerfile.cuda",
        "dockerfile.cpu",
        "containerfile",
        "base_image",
        "base-image",
        ".dockerignore",
    }
)

_FROM_RE = re.compile(
    r"^\s*FROM\s+(?:--\S+\s+)*(\S+)",
    re.MULTILINE | re.IGNORECASE,
)
_ARG_RE = re.compile(
    r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$",
    re.MULTILINE,
)
_SHA256_SUFFIX = re.compile(r"@sha256:[a-fA-F0-9]{64}$")
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Recipe files required in a docker build context (non-sealed bulk + locks).
_RECIPE_BUILD_COPY: Final[tuple[str, ...]] = (
    "requirements.lock",
    "requirements-torch-cpu.lock",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src",
    ".rules",
    "tests",
    "docs",
)


class BuildContextError(ValueError):
    """Base class for build-context / seal-constraint violations."""

    def __init__(self, message: str, *, path: str = "", reason_code: str = "") -> None:
        super().__init__(message)
        self.path = path
        self.reason_code = reason_code


class SealCollisionError(BuildContextError):
    """Miner tree path collides with a sealed surface."""


class ForbiddenMinerArtifactError(BuildContextError):
    """Miner supplied Dockerfile, base image pin, or other BASE-owned artifact."""


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """Result of staging a build context (path + miner subtree name)."""

    root: Path
    miner_subtree: str = MINER_SUBTREE


def sealed_file_sha256(path: Path) -> str:
    """Return hex SHA-256 of file bytes (for seal-intact checks)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _is_sealed_relative(rel: str) -> bool:
    """True if ``rel`` (posix, relative) collides with a sealed surface."""
    rel_n = _norm_rel(rel)
    if not rel_n or rel_n == ".":
        return False
    for sealed in SEALED_RELATIVE_PATHS:
        sealed_n = _norm_rel(sealed)
        if sealed_n.endswith("/"):
            prefix = sealed_n
            if rel_n == prefix.rstrip("/") or rel_n.startswith(prefix):
                return True
            # bare dir name match (e.g. miner ships ".rules/foo")
            if rel_n.startswith(prefix.rstrip("/") + "/"):
                return True
        else:
            if rel_n == sealed_n:
                return True
            # basename collision for known sealed leaf names (harness.py at any depth)
            sealed_name = Path(sealed_n).name
            if Path(rel_n).name == sealed_name and sealed_name.endswith(".py"):
                # Only treat as collision for the known sealed module basenames.
                return True
            if rel_n.endswith("/" + sealed_n) or rel_n.endswith(sealed_n):
                if rel_n == sealed_n or rel_n.endswith("/" + sealed_n):
                    return True
    # Directory markers / sealed stems that must never appear in miner trees.
    parts = rel_n.split("/")
    if ".rules" in parts or "rules" in parts and parts[0] in {".rules", "rules"}:
        return True
    sealed_basenames = {
        "harness.py",
        "loader.py",
        "llm_gate.py",
        "smoke_train.py",
        "gpu_train.py",
        "tiny_1m.py",
    }
    if "prism_recipe" in parts and any(p in sealed_basenames for p in parts):
        return True
    for part in parts:
        stem = part[:-3] if part.endswith(".py") else part
        if part in SEALED_PATH_STEMS or stem in SEALED_PATH_STEMS:
            return True
    return False


def _forbidden_artifact_name(name: str) -> bool:
    lower = name.lower()
    if lower in _FORBIDDEN_MINER_NAMES:
        return True
    if lower.startswith("dockerfile"):
        return True
    if lower in {"base_image", "base-image", "base.image"}:
        return True
    return False


def validate_miner_tree(miner_root: Path | str) -> None:
    """Reject sealed-path collisions and BASE-owned artifacts in a miner tree.

    Raises:
        SealCollisionError: named path collides with a sealed surface.
        ForbiddenMinerArtifactError: Dockerfile / base-image / similar.
        BuildContextError: miner_root missing or not a directory.
    """
    root = Path(miner_root)
    if not root.is_dir():
        raise BuildContextError(
            f"miner_root is not a directory: {root}",
            path=str(root),
            reason_code="missing_miner_root",
        )
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            # Still flag sealed directory names if present as empty dirs.
            if path.is_dir():
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if _is_sealed_relative(rel) or path.name == ".rules":
                    raise SealCollisionError(
                        f"miner tree collides with sealed path: {rel}",
                        path=rel,
                        reason_code="seal_collision",
                    )
            continue
        rel = path.relative_to(root).as_posix()
        name = path.name
        if _forbidden_artifact_name(name):
            raise ForbiddenMinerArtifactError(
                f"miner must not supply BASE-owned artifact: {rel}",
                path=rel,
                reason_code="forbidden_artifact",
            )
        if _is_sealed_relative(rel) or _is_sealed_relative(name):
            raise SealCollisionError(
                f"miner tree collides with sealed path: {rel}",
                path=rel,
                reason_code="seal_collision",
            )


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
    for _ in range(8):
        if cur == prev:
            break
        prev = cur
        cur = _VAR_RE.sub(repl, cur)
    return cur


def _registry_host(image_ref: str) -> str:
    """Best-effort registry host from a FROM image reference (pre-digest)."""
    ref = image_ref.split("@", 1)[0]
    # drop tag
    if ":" in ref.split("/")[-1]:
        ref = ref.rsplit(":", 1)[0]
    first = ref.split("/", 1)[0]
    if "." in first or first == "localhost" or first.endswith(":5000"):
        return first.lower()
    # short name → docker hub
    return "docker.io"


def validate_dockerfile_network_policy(text: str, *, path: str = "Dockerfile") -> None:
    """Require network-policy comments and digest-pinned FROM images only.

    Build network is denied except pinned registries (see
    ``ALLOWED_BUILD_REGISTRIES``). Enforcement here is static validation of the
    Dockerfile text used for BASE builds; orchestrators should also pass
    equivalent registry allowlists to the builder.
    """
    lowered = text.lower()
    if "network" not in lowered:
        raise BuildContextError(
            f"{path}: must document build network policy (deny except pinned registries)",
            path=path,
            reason_code="missing_network_policy_comment",
        )
    if "registry" not in lowered and "registries" not in lowered:
        raise BuildContextError(
            f"{path}: network policy comment must mention pinned registries",
            path=path,
            reason_code="missing_registry_policy_comment",
        )

    defaults = _arg_defaults(text)
    images: list[str] = []
    for match in _FROM_RE.finditer(text):
        raw = match.group(1)
        if raw.upper() == "SCRATCH":
            continue
        expanded = _expand_vars(raw, defaults)
        expanded = _expand_vars(expanded, defaults)
        images.append(expanded)

    if not images:
        raise BuildContextError(
            f"{path}: no FROM lines found",
            path=path,
            reason_code="missing_from",
        )

    for img in images:
        if not _SHA256_SUFFIX.search(img):
            raise BuildContextError(
                f"{path}: unpinned base image (require @sha256:<64-hex>): {img}",
                path=path,
                reason_code="unpinned_base",
            )
        host = _registry_host(img)
        # docker hub short names normalize to docker.io
        if host not in ALLOWED_BUILD_REGISTRIES:
            raise BuildContextError(
                f"{path}: base image registry {host!r} not in allowlist "
                f"{sorted(ALLOWED_BUILD_REGISTRIES)}: {img}",
                path=path,
                reason_code="registry_denied",
            )


def _copy_path(src: Path, dest: Path) -> None:
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=False)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def assemble_build_context(
    *,
    recipe_root: Path | str,
    miner_root: Path | str,
    dest: Path | str,
) -> Path:
    """Stage a docker build context: miner first under ``miner/``, sealed after.

    Order is load-bearing:
    1. Validate miner tree (collisions / forbidden artifacts).
    2. Create empty dest.
    3. Copy miner → ``dest/miner/``.
    4. Copy recipe build inputs (locks, src, .rules, …) **after** miner so sealed
       surfaces always come from the recipe tree.

    Returns:
        Path to the assembled context root (same as ``dest``).
    """
    recipe = Path(recipe_root).resolve()
    miner = Path(miner_root).resolve()
    out = Path(dest)

    validate_miner_tree(miner)

    if out.exists():
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()
    out.mkdir(parents=True)

    # 1) Miner subtree only — never at context root sealed locations.
    miner_dest = out / MINER_SUBTREE
    shutil.copytree(miner, miner_dest, symlinks=False)

    # 2) Recipe surfaces after miner (sealed win by construction).
    for name in _RECIPE_BUILD_COPY:
        src = recipe / name
        if not src.exists():
            # torch CPU lock is CPU-only; optional for cuda-oriented staging.
            if name == "requirements-torch-cpu.lock":
                continue
            raise BuildContextError(
                f"recipe build input missing: {name}",
                path=name,
                reason_code="missing_recipe_input",
            )
        _copy_path(src, out / name)

    # Integrity: sealed digests must match recipe originals.
    for rel in SEALED_RELATIVE_PATHS:
        if rel.endswith("/"):
            continue
        sealed = out / rel
        original = recipe / rel
        if not original.is_file():
            raise BuildContextError(
                f"sealed original missing from recipe: {rel}",
                path=rel,
                reason_code="missing_sealed_original",
            )
        if not sealed.is_file():
            raise BuildContextError(
                f"sealed path missing from assembled context: {rel}",
                path=rel,
                reason_code="missing_sealed_copy",
            )
        if sealed_file_sha256(sealed) != sealed_file_sha256(original):
            raise SealCollisionError(
                f"sealed surface digest mismatch after assemble: {rel}",
                path=rel,
                reason_code="seal_digest_mismatch",
            )

    return out


def iter_sealed_files(recipe_root: Path | str) -> Iterable[Path]:
    """Yield existing sealed file paths under the recipe root."""
    root = Path(recipe_root)
    seen: set[Path] = set()
    for rel in sorted(SEALED_RELATIVE_PATHS):
        if rel.endswith("/"):
            directory = root / rel.rstrip("/")
            if directory.is_dir():
                for path in sorted(directory.rglob("*")):
                    if path.is_file() and path not in seen:
                        seen.add(path)
                        yield path
            continue
        path = root / rel
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path
