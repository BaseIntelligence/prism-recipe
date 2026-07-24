"""Recipe configuration pins (prod egalitarian window + smoke overrides).

Egalitarian contract (VAL-RECIPE-002 / VAL-RECIPE-008):
- Fixed HuggingFace FineWeb-Edu dataset id + immutable revision (commit SHA).
- Fixed global token start offset (EQUAL_OFFSET) shared by every architecture.
- Production token budget TOKEN_BUDGET_PROD = 2_500_000_000 (single-pass / one epoch).
- Smoke/CI may shrink only the budget via PRISM_RECIPE_TOKEN_BUDGET; offset identity
  never changes.

Workers load HF (or cached shards) themselves. A live master FineWeb mount is
NOT required for this product path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Prod egalitarian FineWeb-Edu window pins
# ---------------------------------------------------------------------------

# HuggingFace dataset identity (same for all architectures).
PROD_DATASET_ID = "HuggingFaceFW/fineweb-edu"

# Immutable commit pin (NOT a moving branch/tag). Matches platform prism
# ``FINEWEB_EDU_PIN_SHA`` so recipe and classic locked plane share provenance.
PROD_DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

# Global token start offset into the streaming train split. Chosen as 0 so the
# egalitarian window begins at the head of the pinned revision stream. Every
# architecture uses this same offset; architectures must not introduce their own.
PROD_TOKEN_START_OFFSET = 0
EQUAL_OFFSET = PROD_TOKEN_START_OFFSET  # VAL alias

# Production single-pass budget (2.5B tokens). Long GPU train is NOT required
# for engineering smoke; the constant is the prod pin.
PROD_TOKEN_BUDGET = 2_500_000_000
TOKEN_BUDGET_PROD = PROD_TOKEN_BUDGET  # VAL alias

# Single-pass / one epoch (no multi-epoch rescan of the window).
PROD_EPOCHS = 1

# HF streaming split / config used when workers load FineWeb themselves.
PROD_DATASET_CONFIG = "default"
PROD_DATASET_SPLIT = "train"

# Env override for smoke / unit tests (does not change offset pin identity).
SMOKE_TOKEN_BUDGET_ENV = "PRISM_RECIPE_TOKEN_BUDGET"

# OpenRouter pre-train rules gate pins (VAL-RECIPE-003 / VAL-RECIPE-004).
# Model is request-pinned; miners supply OPENROUTER_API_KEY only (never bake keys).
DEFAULT_OPENROUTER_MODEL = "x-ai/grok-4.5"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CHAT_URL = OPENROUTER_API_URL
OPENROUTER_HTTP_TIMEOUT_SECONDS = 120.0
# Bound architecture/training source size embedded in the gate prompt.
OPENROUTER_MAX_SOURCE_CHARS = 48_000
OPENROUTER_PROMPT_VERSION = "prism-recipe-rules-gate-v1"


@dataclass(frozen=True, slots=True)
class DataWindowPin:
    """Immutable egalitarian data window identity."""

    dataset_id: str
    dataset_revision: str
    token_start_offset: int
    token_budget: int
    epochs: int = 1
    dataset_config: str = PROD_DATASET_CONFIG
    dataset_split: str = PROD_DATASET_SPLIT

    def identity_tuple(self) -> tuple[str, str, int]:
        """Pin identity without budget (smoke may shrink budget only)."""
        return (self.dataset_id, self.dataset_revision, self.token_start_offset)

    def validate(self) -> None:
        """Raise ValueError if the pin violates the single-pass egalitarian contract."""
        if self.epochs != 1:
            raise ValueError(
                f"egalitarian window requires single-pass epochs=1, got epochs={self.epochs}"
            )
        if self.token_start_offset < 0:
            raise ValueError("token_start_offset must be >= 0")
        if self.token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        if not self.dataset_revision:
            raise ValueError("dataset_revision is required")


def prod_data_window() -> DataWindowPin:
    """Return the production egalitarian window pin."""
    pin = DataWindowPin(
        dataset_id=PROD_DATASET_ID,
        dataset_revision=PROD_DATASET_REVISION,
        token_start_offset=PROD_TOKEN_START_OFFSET,
        token_budget=PROD_TOKEN_BUDGET,
        epochs=PROD_EPOCHS,
        dataset_config=PROD_DATASET_CONFIG,
        dataset_split=PROD_DATASET_SPLIT,
    )
    pin.validate()
    return pin


def resolve_token_budget(*, default: int | None = None) -> int:
    """Resolve token budget; smoke path may override via PRISM_RECIPE_TOKEN_BUDGET."""
    raw = os.environ.get(SMOKE_TOKEN_BUDGET_ENV)
    if raw is None or raw.strip() == "":
        return default if default is not None else PROD_TOKEN_BUDGET
    value = int(raw)
    if value < 1:
        raise ValueError(f"{SMOKE_TOKEN_BUDGET_ENV} must be >= 1, got {value}")
    return value
