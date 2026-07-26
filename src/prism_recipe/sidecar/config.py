"""Sidecar runtime configuration (env + explicit)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from prism_recipe.attestation_secret import ATTESTATION_HMAC_KEY_PATH
from prism_recipe.sidecar.errors import SidecarError

Variant = Literal["cpu", "cuda"]

ENV_POD_ID: Final[str] = "PRISM_SIDECAR_POD_ID"
ENV_DIGEST: Final[str] = "PRISM_SIDECAR_IMAGE_DIGEST"
ENV_VARIANT: Final[str] = "PRISM_RECIPE_VARIANT"
ENV_SECRET_PATH: Final[str] = "PRISM_SIDECAR_SECRET_PATH"
ENV_BASE_URL: Final[str] = "PRISM_SIDECAR_BASE_URL"
ENV_RETRY_BUDGET: Final[str] = "PRISM_SIDECAR_RETRY_BUDGET"
ENV_INTERVAL_COUNT: Final[str] = "PRISM_SIDECAR_INTERVAL_COUNT"
ENV_INTERVAL_MIN_S: Final[str] = "PRISM_SIDECAR_INTERVAL_MIN_S"
ENV_INTERVAL_MAX_S: Final[str] = "PRISM_SIDECAR_INTERVAL_MAX_S"

DEFAULT_RETRY_BUDGET: Final[int] = 3
DEFAULT_INTERVAL_COUNT: Final[int] = 2
DEFAULT_INTERVAL_MIN_S: Final[float] = 30.0
DEFAULT_INTERVAL_MAX_S: Final[float] = 120.0


@dataclass(frozen=True, slots=True)
class SidecarConfig:
    """Immutable sidecar identity + schedule knobs."""

    pod_id: str
    digest: str
    variant: Variant
    secret_path: Path
    root: Path | None = None
    retry_budget: int = DEFAULT_RETRY_BUDGET
    interval_count: int = DEFAULT_INTERVAL_COUNT
    interval_min_s: float = DEFAULT_INTERVAL_MIN_S
    interval_max_s: float = DEFAULT_INTERVAL_MAX_S
    base_url: str | None = None

    def __post_init__(self) -> None:
        if not self.pod_id.strip():
            raise SidecarError("pod_id must be non-empty")
        if not self.digest.strip():
            raise SidecarError("digest must be non-empty")
        if self.variant not in {"cpu", "cuda"}:
            raise SidecarError(f"variant must be cpu|cuda, got {self.variant!r}")
        if self.retry_budget < 1:
            raise SidecarError("retry_budget must be >= 1")
        if self.interval_count < 0:
            raise SidecarError("interval_count must be >= 0")
        if self.interval_min_s < 0 or self.interval_max_s < self.interval_min_s:
            raise SidecarError("interval bounds invalid")
        object.__setattr__(self, "secret_path", Path(self.secret_path))

    @staticmethod
    def default_secret_path() -> Path:
        """Canonical mech-6 baked key path."""
        return Path(ATTESTATION_HMAC_KEY_PATH)

    @classmethod
    def from_env(cls) -> SidecarConfig:
        """Load config from process environment (container entry)."""
        pod_id = os.environ.get(ENV_POD_ID, "").strip()
        digest = os.environ.get(ENV_DIGEST, "").strip()
        variant_raw = os.environ.get(ENV_VARIANT, "cpu").strip().lower() or "cpu"
        if variant_raw in {"cpu", "cpu-smoke", "smoke"}:
            variant: Variant = "cpu"
        elif variant_raw in {"cuda", "gpu", "gpu-short"}:
            variant = "cuda"
        else:
            raise SidecarError(f"unknown variant env: {variant_raw!r}")
        secret_raw = os.environ.get(ENV_SECRET_PATH, "").strip()
        secret_path = Path(secret_raw) if secret_raw else cls.default_secret_path()
        base_url = os.environ.get(ENV_BASE_URL, "").strip() or None
        return cls(
            pod_id=pod_id or _require_env(ENV_POD_ID),
            digest=digest or _require_env(ENV_DIGEST),
            variant=variant,
            secret_path=secret_path,
            retry_budget=_env_int(ENV_RETRY_BUDGET, DEFAULT_RETRY_BUDGET),
            interval_count=_env_int(ENV_INTERVAL_COUNT, DEFAULT_INTERVAL_COUNT),
            interval_min_s=_env_float(ENV_INTERVAL_MIN_S, DEFAULT_INTERVAL_MIN_S),
            interval_max_s=_env_float(ENV_INTERVAL_MAX_S, DEFAULT_INTERVAL_MAX_S),
            base_url=base_url,
        )


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SidecarError(f"missing required env {name}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SidecarError(f"{name} must be int, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SidecarError(f"{name} must be float, got {raw!r}") from exc
