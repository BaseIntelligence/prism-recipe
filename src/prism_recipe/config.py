"""Recipe configuration pins (prod egalitarian window + smoke overrides).

Full loader and train loop land in later features. This module only documents
and exposes the contract constants so other packages can import a stable pin.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Prod egalitarian FineWeb-Edu window (VAL-RECIPE-008 pin).
# Identity: same dataset revision + global token start for every architecture.
PROD_DATASET_ID = "HuggingFaceFW/fineweb-edu"
PROD_DATASET_REVISION = "main"  # pin tightened when loader is implemented
PROD_TOKEN_START_OFFSET = 0  # documented fixed global start; shared by all arches
PROD_TOKEN_BUDGET = 2_500_000_000  # single-pass / one epoch
PROD_EPOCHS = 1

# Env override for smoke / unit tests (does not change offset pin identity).
SMOKE_TOKEN_BUDGET_ENV = "PRISM_RECIPE_TOKEN_BUDGET"

# OpenRouter gate (stub pin; LLM client implemented in a later feature).
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True, slots=True)
class DataWindowPin:
    """Immutable egalitarian data window identity."""

    dataset_id: str
    dataset_revision: str
    token_start_offset: int
    token_budget: int
    epochs: int = 1

    def identity_tuple(self) -> tuple[str, str, int]:
        """Pin identity without budget (smoke may shrink budget only)."""
        return (self.dataset_id, self.dataset_revision, self.token_start_offset)


def prod_data_window() -> DataWindowPin:
    """Return the production egalitarian window pin."""
    return DataWindowPin(
        dataset_id=PROD_DATASET_ID,
        dataset_revision=PROD_DATASET_REVISION,
        token_start_offset=PROD_TOKEN_START_OFFSET,
        token_budget=PROD_TOKEN_BUDGET,
        epochs=PROD_EPOCHS,
    )


def resolve_token_budget(*, default: int | None = None) -> int:
    """Resolve token budget; smoke path may override via PRISM_RECIPE_TOKEN_BUDGET."""
    raw = os.environ.get(SMOKE_TOKEN_BUDGET_ENV)
    if raw is None or raw.strip() == "":
        return default if default is not None else PROD_TOKEN_BUDGET
    return int(raw)
