"""Image-sealed architecture package (no mid-run miner mutation path)."""

from prism_recipe.arch.tiny_1m import (
    MODEL_DIM,
    MODEL_HEADS,
    MODEL_LAYERS,
    MODEL_MLP_RATIO,
    MODEL_VOCAB_SIZE,
    TinyDecoderLM,
    build_tiny_1m,
    count_parameters,
)

__all__ = [
    "MODEL_DIM",
    "MODEL_HEADS",
    "MODEL_LAYERS",
    "MODEL_MLP_RATIO",
    "MODEL_VOCAB_SIZE",
    "TinyDecoderLM",
    "build_tiny_1m",
    "count_parameters",
]
