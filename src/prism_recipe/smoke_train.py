"""Offline tiny-1m smoke train (VAL-RECIPE-006).

Runs the sealed harness path end-to-end without network:
LLM gate (injectable) → egalitarian loader on a tiny fixture stream →
few AdamW steps on transformer-tiny-1m under a small ``token_budget``.

Torch is required only for the actual train loop (``train`` extra /
image layers). Pure planning / param checks stay import-light.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from prism_recipe.arch.tiny_1m import (
    MAX_PARAMS,
    MODEL_VOCAB_SIZE,
    build_tiny_1m,
    count_parameters,
)
from prism_recipe.config import resolve_token_budget
from prism_recipe.loader import EgalitarianFineWebLoader, TextDocument, build_loader
from prism_recipe.llm_gate import GateResult, run_rules_gate
from prism_recipe.sealed_surface import (
    ensure_execution_sealed,
    read_sealed_sources,
    resolve_repo_root,
    train_module_relative_path,
)

# Image-bundled sealed architecture / training sources (no miner swap).
# Canonical sealing lives in sealed_surface (M19: variant-aware train module).

_REPO_ROOT = resolve_repo_root()
SEALED_ARCH_PATH = _REPO_ROOT / "src" / "prism_recipe" / "arch" / "tiny_1m.py"
SEALED_TRAIN_PATH = _REPO_ROOT / train_module_relative_path("cpu")

DEFAULT_SMOKE_TOKEN_BUDGET = 256
DEFAULT_SMOKE_SEQ_LEN = 32
DEFAULT_SMOKE_BATCH_SIZE = 4
DEFAULT_SMOKE_MAX_STEPS = 3
DEFAULT_SMOKE_LR = 5e-3
ARCH_NAME = "transformer-tiny-1m"


@dataclass(frozen=True, slots=True)
class SmokeTrainResult:
    """Structured smoke train outcome for harness / CLI / tests."""

    ok: bool
    stage: str
    message: str
    param_count: int = 0
    tokens_seen: int = 0
    steps: int = 0
    final_loss: float | None = None
    token_budget: int = 0
    architecture: str = ARCH_NAME
    device: str = "cpu"
    gate: GateResult | None = None
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "stage": self.stage,
            "message": self.message,
            "architecture": self.architecture,
            "param_count": self.param_count,
            "tokens_seen": self.tokens_seen,
            "steps": self.steps,
            "final_loss": self.final_loss,
            "token_budget": self.token_budget,
            "device": self.device,
            "mid_run_miner_mutation": False,
        }
        if self.gate is not None:
            payload["gate"] = self.gate.as_dict()
            payload["attestation"] = self.gate.attestation_metadata()
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def tiny_fixture_documents(*, n_docs: int = 32, words_per_doc: int = 48) -> list[TextDocument]:
    """Deterministic offline FineWeb-shaped documents (no HF network)."""
    docs: list[TextDocument] = []
    # Repeatable vocabulary of short tokens.
    lexicon = [f"w{i:03d}" for i in range(64)]
    for i in range(n_docs):
        words = [lexicon[(i * 7 + j) % len(lexicon)] for j in range(words_per_doc)]
        docs.append(
            {
                "text": " ".join(words),
                "id": f"fixture-{i:04d}",
            }
        )
    return docs


def fixture_encode(text: str, *, vocab_size: int = MODEL_VOCAB_SIZE) -> list[int]:
    """Map whitespace words to stable ids in [0, vocab_size)."""
    if not text or not text.strip():
        return []
    out: list[int] = []
    for word in text.split():
        # Stable positive id independent of PYTHONHASHSEED.
        h = 2166136261
        for ch in word.encode("utf-8"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        out.append((h % (vocab_size - 1)) + 1)
    return out


def _next_token_loss(logits, tokens):
    import torch.nn.functional as F

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    vocab = logits.shape[-1]
    predictions = logits[:, :-1, :].reshape(-1, vocab)
    targets = tokens[:, 1:].reshape(-1) % vocab
    return F.cross_entropy(predictions, targets)


def _batches_from_tokens(
    token_ids: Sequence[int],
    *,
    seq_len: int,
    batch_size: int,
    max_steps: int,
) -> Iterable[list[list[int]]]:
    """Yield dense token batches from a flat token stream."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    need = seq_len * batch_size
    flat = list(token_ids)
    if len(flat) < need:
        # Repeat fixture tokens to fill at least one batch (offline smoke).
        if not flat:
            flat = [1] * need
        reps = (need // len(flat)) + 2
        flat = (flat * reps)[: max(need * max_steps, need)]
    cursor = 0
    for _ in range(max_steps):
        chunk = flat[cursor : cursor + need]
        if len(chunk) < need:
            break
        batch = [chunk[i * seq_len : (i + 1) * seq_len] for i in range(batch_size)]
        yield batch
        cursor += need


def run_smoke_train(
    *,
    token_budget: int | None = None,
    document_source: Iterable[TextDocument] | None = None,
    gate: GateResult | None = None,
    skip_gate: bool | None = None,
    max_steps: int = DEFAULT_SMOKE_MAX_STEPS,
    seq_len: int = DEFAULT_SMOKE_SEQ_LEN,
    batch_size: int = DEFAULT_SMOKE_BATCH_SIZE,
    learning_rate: float = DEFAULT_SMOKE_LR,
    device: str | None = None,
    seed: int = 0,
) -> SmokeTrainResult:
    """End-to-end offline smoke: gate → fixture loader → tiny-1m train steps.

    Parameters
    ----------
    gate:
        Optional precomputed gate result (tests inject pass/fail).
    skip_gate:
        When True, synthesize a local pass gate without calling OpenRouter.
        Defaults from env ``PRISM_RECIPE_SMOKE_SKIP_GATE=1`` for offline CI.
    document_source:
        Optional mocked HF stream; defaults to :func:`tiny_fixture_documents`.
    """
    budget = (
        int(token_budget)
        if token_budget is not None
        else resolve_token_budget(default=DEFAULT_SMOKE_TOKEN_BUDGET)
    )
    if budget < 1:
        return SmokeTrainResult(
            ok=False,
            stage="config",
            message="invalid_token_budget",
            token_budget=budget,
        )

    # Resolve skip_gate default from env for offline image smoke.
    if skip_gate is None:
        skip_gate = os.environ.get("PRISM_RECIPE_SMOKE_SKIP_GATE", "").strip() in {
            "1",
            "true",
            "yes",
        }

    ensure_execution_sealed(variant="cpu")
    arch_src, train_src = read_sealed_sources(variant="cpu")
    if gate is None:
        if skip_gate:
            from prism_recipe.llm_gate import rules_digest
            from datetime import datetime, timezone

            gate = GateResult(
                ok=True,
                reason="smoke_skip_gate",
                rules_digest=rules_digest(),
                model="smoke-local",
                checked_at=datetime.now(timezone.utc).isoformat(),
                decision="pass",
                prompt_hash="",
                reason_codes=("smoke_skip_gate",),
            )
        else:
            gate = run_rules_gate(
                architecture_source=arch_src,
                training_source=train_src,
            )

    if not gate.ok:
        return SmokeTrainResult(
            ok=False,
            stage="llm_gate",
            message=gate.reason,
            token_budget=budget,
            gate=gate,
            metadata={"mid_run_miner_mutation": False},
        )

    docs = list(document_source) if document_source is not None else tiny_fixture_documents()
    loader: EgalitarianFineWebLoader = build_loader(
        document_source=docs,
        encode_fn=fixture_encode,
        token_budget=budget,
    )
    plan = loader.plan()
    token_ids = list(loader.iter_tokens())
    tokens_collected = len(token_ids)
    # Single-pass budget stop: never exceed budget.
    if tokens_collected > budget:
        return SmokeTrainResult(
            ok=False,
            stage="loader",
            message="budget_exceeded",
            tokens_seen=tokens_collected,
            token_budget=budget,
            gate=gate,
            metadata={"plan": plan.as_dict()},
        )

    try:
        import torch
    except ImportError:
        return SmokeTrainResult(
            ok=False,
            stage="train",
            message="torch_not_installed",
            tokens_seen=tokens_collected,
            token_budget=budget,
            gate=gate,
            metadata={
                "plan": plan.as_dict(),
                "hint": "pip install 'prism-recipe[train]' or use the recipe image",
            },
        )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tiny_1m(device=resolved_device, seed=seed)
    n_params = count_parameters(model)
    if n_params > MAX_PARAMS:
        return SmokeTrainResult(
            ok=False,
            stage="architecture",
            message="param_count_exceeds_cap",
            param_count=n_params,
            tokens_seen=tokens_collected,
            token_budget=budget,
            gate=gate,
            device=resolved_device,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    steps = 0
    last_loss: float | None = None
    for batch in _batches_from_tokens(
        token_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        max_steps=max_steps,
    ):
        tensor = torch.tensor(batch, dtype=torch.long, device=resolved_device)
        optimizer.zero_grad(set_to_none=True)
        loss = _next_token_loss(model(tensor), tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        last_loss = float(loss.detach().cpu().item())

    if steps < 1:
        return SmokeTrainResult(
            ok=False,
            stage="train",
            message="no_train_steps",
            param_count=n_params,
            tokens_seen=tokens_collected,
            token_budget=budget,
            gate=gate,
            device=resolved_device,
            metadata={"plan": plan.as_dict()},
        )

    return SmokeTrainResult(
        ok=True,
        stage="smoke_train",
        message="smoke_train_ok",
        param_count=n_params,
        tokens_seen=tokens_collected,
        steps=steps,
        final_loss=last_loss,
        token_budget=budget,
        architecture=ARCH_NAME,
        device=resolved_device,
        gate=gate,
        metadata={
            "plan": plan.as_dict(),
            "max_params": MAX_PARAMS,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "mid_run_miner_mutation": False,
            "sealed_arch_path": str(SEALED_ARCH_PATH.name),
            "image_contents": ["rules", "harness", "loader", "llm_gate", "tiny_1m"],
        },
    )


__all__ = [
    "ARCH_NAME",
    "DEFAULT_SMOKE_TOKEN_BUDGET",
    "SmokeTrainResult",
    "fixture_encode",
    "read_sealed_sources",
    "run_smoke_train",
    "tiny_fixture_documents",
]
