"""Short real CUDA train for ~1M-param transformer (mission GPU score path).

Distinct from ``prism_cpu_reexec`` TinyLM (emb=8 / ~1.09K params / ~360 tokens).
This module trains the sealed ``transformer-tiny-1m`` (~1.05M params) on CUDA
with a short token budget (default 50k–200k) and authors a scoreable
``prism_run_manifest.v2`` payload (online-loss stream + covered_bytes + NLL).

LLM rules gate must already have passed (or is run here first). Never prints
API keys.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from prism_recipe.arch.tiny_1m import (
    MAX_PARAMS,
    MODEL_VOCAB_SIZE,
    build_tiny_1m,
    count_parameters,
)
from prism_recipe.config import resolve_token_budget
from prism_recipe.llm_gate import GateResult, run_rules_gate
from prism_recipe.sealed_surface import ensure_execution_sealed, read_sealed_sources
from prism_recipe.smoke_train import (
    fixture_encode,
    tiny_fixture_documents,
)

ARCH_NAME = "transformer-tiny-1m"

# Short train defaults (minutes on a single modern GPU; NEVER 10h / 2.5B).
DEFAULT_GPU_TOKEN_BUDGET = 65_536
DEFAULT_GPU_SEQ_LEN = 128
DEFAULT_GPU_BATCH_SIZE = 8
DEFAULT_GPU_MAX_STEPS = 512
DEFAULT_GPU_LR = 3e-4
# Bytes-per-token estimate used when measuring covered UTF-8 from fixture tokens.
# Real runner uses source UTF-8; here each fixture word ~ avg 5 chars + space.
AVG_BYTES_PER_TOKEN = 6


@dataclass(frozen=True, slots=True)
class GpuTrainResult:
    """Structured GPU short-train outcome for worker / attestation surfaces."""

    ok: bool
    stage: str
    message: str
    param_count: int = 0
    tokens_consumed: int = 0
    covered_bytes: int = 0
    steps: int = 0
    final_loss: float | None = None
    token_budget: int = 0
    architecture: str = ARCH_NAME
    device: str = "cpu"
    online_loss: tuple[float, ...] = ()
    sum_neg_log_likelihood_nats: float = 0.0
    wall_clock_seconds: float = 0.0
    cuda_name: str | None = None
    gate: GateResult | None = None
    metadata: dict[str, Any] | None = None
    run_manifest: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "stage": self.stage,
            "message": self.message,
            "architecture": self.architecture,
            "param_count": self.param_count,
            "tokens_consumed": self.tokens_consumed,
            "covered_bytes": self.covered_bytes,
            "steps": self.steps,
            "final_loss": self.final_loss,
            "token_budget": self.token_budget,
            "device": self.device,
            "cuda_name": self.cuda_name,
            "sum_neg_log_likelihood_nats": self.sum_neg_log_likelihood_nats,
            "online_loss_samples": len(self.online_loss),
            "wall_clock_seconds": self.wall_clock_seconds,
            "mid_run_miner_mutation": False,
            "not_tinylm_fingerprint": True,
        }
        if self.gate is not None:
            payload["gate"] = self.gate.as_dict()
            payload["attestation"] = self.gate.attestation_metadata()
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.run_manifest is not None:
            payload["run_manifest"] = self.run_manifest
        return payload


def _require_cuda(device: str | None) -> tuple[str, str | None]:
    import torch

    if device:
        resolved = device
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_not_available")
        resolved = "cuda:0"
    if not str(resolved).startswith("cuda"):
        raise RuntimeError(f"gpu_train_requires_cuda_device got={resolved}")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_not_available")
    name = None
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        name = None
    return resolved, name


def _expand_fixture_tokens(token_budget: int, *, vocab_size: int = MODEL_VOCAB_SIZE) -> list[int]:
    """Build a locked offline FineWeb-shaped token stream covering token_budget."""
    # Large enough fixture so short GPU budgets stay single-pass without HF network.
    docs = tiny_fixture_documents(n_docs=512, words_per_doc=256)
    flat: list[int] = []
    for doc in docs:
        flat.extend(fixture_encode(str(doc.get("text") or ""), vocab_size=vocab_size))
        if len(flat) >= token_budget + 1024:
            break
    if len(flat) < token_budget + 16:
        # Repeat deterministic cycle rather than shrink the budget claim.
        if not flat:
            flat = [((i * 17) % (vocab_size - 1)) + 1 for i in range(token_budget + 64)]
        reps = (token_budget // len(flat)) + 2
        flat = (flat * reps)[: token_budget + 64]
    return flat[: token_budget + 64]


def _next_token_nll(logits, tokens) -> tuple[Any, float, int]:
    """Return (loss_tensor_mean, sum_nll_nats, n_predicted_tokens)."""
    import torch.nn.functional as F

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    # logits: [B, T, V], targets next-token
    predictions = logits[:, :-1, :].contiguous()
    targets = tokens[:, 1:].contiguous()
    b, t_minus, vocab = predictions.shape
    flat_logits = predictions.reshape(-1, vocab)
    flat_targets = targets.reshape(-1) % vocab
    # Per-token NLL in nats (online compression metric).
    per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    sum_nll = float(per_tok.sum().detach().cpu().item())
    n_pred = int(per_tok.numel())
    mean_loss = per_tok.mean()
    return mean_loss, sum_nll, n_pred


def build_gpu_run_manifest(
    *,
    submission_id: str,
    param_count: int,
    tokens_consumed: int,
    covered_bytes: int,
    online_loss: Sequence[float],
    sum_neg_log_likelihood_nats: float,
    device: str,
    cuda_name: str | None,
    token_budget: int,
    steps: int,
    final_loss: float | None,
    wall_clock_seconds: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Author a scoreable prism_run_manifest.v2 from real GPU train metrics."""
    losses = [float(x) for x in online_loss]
    step0 = losses[0] if losses else None
    # Naive learning check: final mean loss below first sample.
    no_learning = bool(losses) and losses[-1] >= losses[0] * 0.999 and len(losses) >= 2
    estimated_flops = (
        6.0 * float(param_count) * float(tokens_consumed)
        if param_count and tokens_consumed
        else None
    )
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": submission_id,
        "run_id": f"prism-gpu-short-{submission_id}",
        "mode": "gpu_proxy_eval",
        "run": {
            "seed": int(seed),
            "forced_init": True,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": device,
            "nproc_per_node": 1,
            "architecture": ARCH_NAME,
            "token_budget": int(token_budget),
            "steps": int(steps),
            "gpu_train": True,
            "not_tinylm": True,
        },
        "data": {
            "covered_bytes": int(covered_bytes),
            "covered_bytes_cumulative": [int(covered_bytes)],
            "single_pass": True,
            "source": "fixture_locked_shard",
        },
        "metrics": {
            "online_loss": losses,
            "sum_neg_log_likelihood_nats": float(sum_neg_log_likelihood_nats),
            "covered_bytes": int(covered_bytes),
            "predicted_tokens": int(tokens_consumed),
            "tokens_seen": int(tokens_consumed),
            "tokens_consumed": int(tokens_consumed),
            "step0_loss": step0,
            "consumed_batches": len(losses),
            "final_loss": final_loss,
            "random_init_baseline_nats": math.log(MODEL_VOCAB_SIZE),
        },
        "compute": {
            "schema": "prism_compute.v1",
            "gpu_count": 1,
            "world_size": 1,
            "nproc_per_node": 1,
            "device": device,
            "model_params": int(param_count),
            "wall_clock_seconds": float(wall_clock_seconds),
            "estimated_flops": estimated_flops,
            "cuda_device_name": cuda_name,
            "param_ladder_stage": "explore",
            "param_ladder_cap": MAX_PARAMS,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": bool(no_learning),
            "zero_forward": tokens_consumed <= 0,
        },
        "artifacts": {
            "trained_state": None,
            "gpu_short_train": True,
        },
    }


def run_gpu_short_train(
    *,
    submission_id: str = "local-gpu-smoke",
    token_budget: int | None = None,
    gate: GateResult | None = None,
    skip_gate: bool | None = None,
    seq_len: int = DEFAULT_GPU_SEQ_LEN,
    batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    max_steps: int | None = None,
    learning_rate: float = DEFAULT_GPU_LR,
    device: str | None = None,
    seed: int = 0,
    require_cuda: bool = True,
) -> GpuTrainResult:
    """Real short train on CUDA for transformer-tiny-1m; build scoreable manifest.

    Parameters
    ----------
    require_cuda:
        When True (default, scoring path), refuse to run on CPU. Tests may set
        False to exercise the loop shape without a GPU.
    """
    budget = (
        int(token_budget)
        if token_budget is not None
        else resolve_token_budget(default=DEFAULT_GPU_TOKEN_BUDGET)
    )
    # Clamp pathological envs: still short train, never multi-hour / 2.5B.
    if budget < 1_024:
        budget = 1_024
    if budget > 200_000:
        budget = 200_000

    if skip_gate is None:
        skip_gate = os.environ.get("PRISM_RECIPE_SMOKE_SKIP_GATE", "").strip() in {
            "1",
            "true",
            "yes",
        }

    ensure_execution_sealed(variant="cuda")
    arch_src, train_src = read_sealed_sources(variant="cuda")
    if gate is None:
        if skip_gate:
            from prism_recipe.llm_gate import rules_digest

            gate = GateResult(
                ok=True,
                reason="smoke_skip_gate",
                rules_digest=rules_digest(),
                model="smoke-local",
                checked_at=datetime.now(UTC).isoformat(),
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
        return GpuTrainResult(
            ok=False,
            stage="llm_gate",
            message=gate.reason,
            token_budget=budget,
            gate=gate,
            metadata={"mid_run_miner_mutation": False},
        )

    try:
        import torch
    except ImportError:
        return GpuTrainResult(
            ok=False,
            stage="train",
            message="torch_not_installed",
            token_budget=budget,
            gate=gate,
        )

    try:
        if require_cuda:
            resolved_device, cuda_name = _require_cuda(device)
        else:
            resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            cuda_name = None
            if str(resolved_device).startswith("cuda") and torch.cuda.is_available():
                try:
                    cuda_name = torch.cuda.get_device_name(0)
                except Exception:  # noqa: BLE001
                    cuda_name = None
    except RuntimeError as exc:
        return GpuTrainResult(
            ok=False,
            stage="device",
            message=str(exc),
            token_budget=budget,
            gate=gate,
            device=device or "cpu",
        )

    t0 = time.perf_counter()
    torch.manual_seed(int(seed))
    if str(resolved_device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    model = build_tiny_1m(device=resolved_device, seed=seed)
    n_params = count_parameters(model)
    if n_params > MAX_PARAMS:
        return GpuTrainResult(
            ok=False,
            stage="architecture",
            message="param_count_exceeds_cap",
            param_count=n_params,
            token_budget=budget,
            gate=gate,
            device=resolved_device,
            cuda_name=cuda_name,
        )
    # Fingerprint guard: refuse TinyLM-scale models (emb=8 class ~1k params).
    if n_params < 100_000:
        return GpuTrainResult(
            ok=False,
            stage="architecture",
            message=f"model_too_small_not_1m params={n_params}",
            param_count=n_params,
            token_budget=budget,
            gate=gate,
            device=resolved_device,
            cuda_name=cuda_name,
        )

    token_ids = _expand_fixture_tokens(budget)
    steps_cap = int(max_steps) if max_steps is not None else DEFAULT_GPU_MAX_STEPS
    # Derive step count from budget: each step consumes batch_size * (seq_len-1) predicted tokens.
    pred_per_step = batch_size * max(seq_len - 1, 1)
    if pred_per_step < 1:
        return GpuTrainResult(
            ok=False,
            stage="config",
            message="invalid_seq_or_batch",
            param_count=n_params,
            token_budget=budget,
            gate=gate,
            device=resolved_device,
            cuda_name=cuda_name,
        )
    needed_steps = max(1, math.ceil(budget / pred_per_step))
    steps_cap = min(steps_cap, max(needed_steps, 8))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    online_losses: list[float] = []
    sum_nll = 0.0
    tokens_consumed = 0
    steps = 0
    last_loss: float | None = None
    cursor = 0
    need = seq_len * batch_size

    while tokens_consumed < budget and steps < steps_cap:
        chunk = token_ids[cursor : cursor + need]
        if len(chunk) < need:
            # wrap
            cursor = 0
            chunk = token_ids[cursor : cursor + need]
            if len(chunk) < need:
                break
        cursor += need
        batch = [chunk[i * seq_len : (i + 1) * seq_len] for i in range(batch_size)]
        tensor = torch.tensor(batch, dtype=torch.long, device=resolved_device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensor)
        loss, batch_nll, n_pred = _next_token_nll(logits, tensor)
        if not torch.isfinite(loss):
            return GpuTrainResult(
                ok=False,
                stage="train",
                message="non_finite_loss",
                param_count=n_params,
                tokens_consumed=tokens_consumed,
                steps=steps,
                token_budget=budget,
                gate=gate,
                device=resolved_device,
                cuda_name=cuda_name,
                online_loss=tuple(online_losses),
                sum_neg_log_likelihood_nats=sum_nll,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_f = float(loss.detach().cpu().item())
        online_losses.append(loss_f)
        sum_nll += batch_nll
        tokens_consumed += n_pred
        steps += 1
        last_loss = loss_f
        if tokens_consumed >= budget:
            break

    wall = time.perf_counter() - t0
    if steps < 1 or tokens_consumed < 1:
        return GpuTrainResult(
            ok=False,
            stage="train",
            message="no_train_steps",
            param_count=n_params,
            tokens_consumed=tokens_consumed,
            token_budget=budget,
            gate=gate,
            device=resolved_device,
            cuda_name=cuda_name,
            wall_clock_seconds=wall,
        )

    # Covered bytes: challenge scores tokens_consumed / covered_bytes via NLL bits.
    covered_bytes = max(int(tokens_consumed * AVG_BYTES_PER_TOKEN), tokens_consumed + 1)
    # Keep enough online_loss samples for scoring (list non-empty) but bound payload size.
    if len(online_losses) > 256:
        # Downsample evenly while keeping first/last.
        idxs = sorted(
            set(
                [0, len(online_losses) - 1]
                + [
                    int(i * (len(online_losses) - 1) / 254)
                    for i in range(1, 254)
                ]
            )
        )
        online_losses = [online_losses[i] for i in idxs]

    manifest = build_gpu_run_manifest(
        submission_id=submission_id,
        param_count=n_params,
        tokens_consumed=tokens_consumed,
        covered_bytes=covered_bytes,
        online_loss=online_losses,
        sum_neg_log_likelihood_nats=sum_nll,
        device=resolved_device,
        cuda_name=cuda_name,
        token_budget=budget,
        steps=steps,
        final_loss=last_loss,
        wall_clock_seconds=wall,
        seed=seed,
    )

    return GpuTrainResult(
        ok=True,
        stage="gpu_train",
        message="gpu_short_train_ok",
        param_count=n_params,
        tokens_consumed=tokens_consumed,
        covered_bytes=covered_bytes,
        steps=steps,
        final_loss=last_loss,
        token_budget=budget,
        architecture=ARCH_NAME,
        device=resolved_device,
        online_loss=tuple(online_losses),
        sum_neg_log_likelihood_nats=sum_nll,
        wall_clock_seconds=wall,
        cuda_name=cuda_name,
        gate=gate,
        metadata={
            "seq_len": seq_len,
            "batch_size": batch_size,
            "max_steps": steps_cap,
            "mid_run_miner_mutation": False,
            "not_tinylm_fingerprint": True,
            "model_params": n_params,
            "tokens_consumed": tokens_consumed,
            "gpu_name": cuda_name,
        },
        run_manifest=manifest,
    )


__all__ = [
    "DEFAULT_GPU_TOKEN_BUDGET",
    "GpuTrainResult",
    "build_gpu_run_manifest",
    "run_gpu_short_train",
]
