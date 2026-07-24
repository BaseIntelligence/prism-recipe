"""Image-sealed transformer-tiny-1m architecture (≤~1.5M params).

Adapted from platform prism ``examples/tiny-1m`` for the recipe product path:
architecture lives **inside** the digest-pinned image (no separate miner ZIP,
no mid-run mutation of this module).

Parameter target with defaults (vocab=4096, dim=128, heads=4, layers=2, mlp=4×):
weight-tied embedding/lm_head → ~1.05M trainable parameters.
"""

from __future__ import annotations

from typing import Any

# Hyperparameters pinned for the sealed smoke / score path.
MODEL_VOCAB_SIZE = 4096
MODEL_DIM = 128
MODEL_HEADS = 4
MODEL_LAYERS = 2
MODEL_MLP_RATIO = 4
DEFAULT_EPS = 1e-6
EMB_INIT_STD = 0.13
# VAL-RECIPE-006 bound: ≤~1.5M params.
MAX_PARAMS = 1_500_000


def count_parameters(module: Any) -> int:
    """Count trainable parameters (works with torch.nn.Module)."""
    total = 0
    for param in module.parameters():
        if getattr(param, "requires_grad", True):
            n = 1
            for dim in param.shape:
                n *= int(dim)
            total += n
    return total


def build_tiny_1m(
    *,
    vocab_size: int = MODEL_VOCAB_SIZE,
    dim: int = MODEL_DIM,
    n_heads: int = MODEL_HEADS,
    n_layers: int = MODEL_LAYERS,
    mlp_ratio: int = MODEL_MLP_RATIO,
    device: str | None = None,
    seed: int | None = 0,
):
    """Construct the sealed TinyDecoderLM (requires torch at call time)."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(int(seed))

    class RMSNorm(nn.Module):
        def __init__(self, d: int, eps: float = DEFAULT_EPS) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            variance = x.mul(x).mean(-1, keepdim=True)
            normed = x * torch.rsqrt(variance + self.eps)
            return normed * self.weight

    class CausalSelfAttention(nn.Module):
        def __init__(self, d: int, heads: int) -> None:
            super().__init__()
            if heads <= 0:
                raise ValueError("n_heads must be positive")
            if d % heads != 0:
                raise ValueError("dim must be divisible by n_heads")
            self.n_heads = heads
            self.head_dim = d // heads
            self.scale = self.head_dim**-0.5
            self.qkv = nn.Linear(d, d * 3, bias=False)
            self.proj = nn.Linear(d, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * self.scale
            causal = torch.triu(
                torch.ones(t, t, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            context = weights @ v
            context = context.transpose(1, 2).contiguous().view(b, t, c)
            return self.proj(context)

    class GatedMLP(nn.Module):
        def __init__(self, d: int, hidden: int) -> None:
            super().__init__()
            self.gate = nn.Linear(d, hidden, bias=False)
            self.up = nn.Linear(d, hidden, bias=False)
            self.down = nn.Linear(hidden, d, bias=False)
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(self.act(self.gate(x)) * self.up(x))

    class TransformerBlock(nn.Module):
        def __init__(self, d: int, heads: int, ratio: int) -> None:
            super().__init__()
            hidden = d * ratio
            self.norm_attn = RMSNorm(d)
            self.attn = CausalSelfAttention(d, heads)
            self.norm_mlp = RMSNorm(d)
            self.mlp = GatedMLP(d, hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.norm_attn(x))
            x = x + self.mlp(self.norm_mlp(x))
            return x

    class TinyDecoderLM(nn.Module):
        """Weight-tied decoder LM used by the recipe smoke / score path."""

        def __init__(
            self,
            vs: int,
            d: int,
            heads: int,
            layers: int,
            ratio: int,
        ) -> None:
            super().__init__()
            if layers <= 0:
                raise ValueError("n_layers must be positive")
            self.vocab_size = vs
            self.dim = d
            self.token_emb = nn.Embedding(vs, d)
            self.blocks = nn.ModuleList(
                [TransformerBlock(d, heads, ratio) for _ in range(layers)]
            )
            self.norm_final = RMSNorm(d)
            self.lm_head = nn.Linear(d, vs, bias=False)
            # Weight tying (shared embedding / head matrix).
            self.lm_head.weight = self.token_emb.weight
            with torch.no_grad():
                self.token_emb.weight.normal_(0.0, EMB_INIT_STD)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            x = self.token_emb(tokens)
            for block in self.blocks:
                x = block(x)
            x = self.norm_final(x)
            return self.lm_head(x)

        def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            logits = self.forward(tokens)
            vocab = logits.shape[-1]
            return F.cross_entropy(
                logits.reshape(-1, vocab),
                targets.reshape(-1) % vocab,
            )

    model = TinyDecoderLM(vocab_size, dim, n_heads, n_layers, mlp_ratio)
    n_params = count_parameters(model)
    if n_params > MAX_PARAMS:
        raise ValueError(
            f"tiny-1m param count {n_params} exceeds MAX_PARAMS={MAX_PARAMS}"
        )
    if device is not None:
        model = model.to(device)
    return model


# Type alias name used by package exports / docs.
TinyDecoderLM = Any  # runtime class is nested inside build_tiny_1m


__all__ = [
    "DEFAULT_EPS",
    "EMB_INIT_STD",
    "MAX_PARAMS",
    "MODEL_DIM",
    "MODEL_HEADS",
    "MODEL_LAYERS",
    "MODEL_MLP_RATIO",
    "MODEL_VOCAB_SIZE",
    "TinyDecoderLM",
    "build_tiny_1m",
    "count_parameters",
]
