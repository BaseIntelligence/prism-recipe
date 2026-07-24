"""Egalitarian HF FineWeb-Edu loader (stub).

Later feature implements deterministic stream over the fixed global token
window. Do not load network datasets from unit tests until that lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from prism_recipe.config import DataWindowPin, prod_data_window, resolve_token_budget


@dataclass(frozen=True, slots=True)
class LoaderPlan:
    """Resolved load plan (pin identity + effective budget)."""

    pin: DataWindowPin
    token_budget: int
    single_pass: bool = True


class EgalitarianFineWebLoader:
    """Stub loader. Raises NotImplementedError until the HF feature lands."""

    def __init__(self, pin: DataWindowPin | None = None) -> None:
        self.pin = pin or prod_data_window()

    def plan(self) -> LoaderPlan:
        budget = resolve_token_budget(default=self.pin.token_budget)
        return LoaderPlan(pin=self.pin, token_budget=budget, single_pass=True)

    def iter_tokens(self) -> Iterator[int]:
        """Yield token ids for the egalitarian window (not implemented)."""
        raise NotImplementedError(
            "Egalitarian FineWeb loader is a stub; implement in "
            "prism-recipe-hf-egalitarian-loader feature."
        )

    def iter_text_shards(self) -> Iterator[str]:
        """Yield raw text shards (not implemented)."""
        raise NotImplementedError(
            "Egalitarian FineWeb loader is a stub; implement in "
            "prism-recipe-hf-egalitarian-loader feature."
        )


def build_loader(pin: DataWindowPin | None = None) -> EgalitarianFineWebLoader:
    """Factory used by harness entrypoints."""
    return EgalitarianFineWebLoader(pin=pin)
