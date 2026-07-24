"""PRISM recipe harness package (image-sealed architecture + training + eval)."""

from prism_recipe.config import (
    EQUAL_OFFSET,
    PROD_DATASET_ID,
    PROD_DATASET_REVISION,
    PROD_TOKEN_BUDGET,
    TOKEN_BUDGET_PROD,
)

__version__ = "0.1.0"

__all__ = [
    "EQUAL_OFFSET",
    "PROD_DATASET_ID",
    "PROD_DATASET_REVISION",
    "PROD_TOKEN_BUDGET",
    "TOKEN_BUDGET_PROD",
    "__version__",
]
