"""Recipe harness entrypoints.

Product model: architecture + training + train loop live **inside** this
image. Miners do not submit a separate architecture ZIP for this product path.
They supply ``OPENROUTER_API_KEY`` and Lium deploy environment only.

There is **no mid-run miner mutation path** — rules, harness, loader, LLM gate,
and sealed tiny-1m architecture are image-bundled and digest-pinned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from prism_recipe.config import prod_data_window, resolve_token_budget
from prism_recipe.loader import build_loader
from prism_recipe.llm_gate import GateResult, run_rules_gate
from prism_recipe.smoke_train import read_sealed_sources, run_smoke_train


def _want_gpu_short() -> bool:
    """True when PRISM_RECIPE_GPU_SHORT=1 (real CUDA ~1M short train score path)."""
    return os.environ.get("PRISM_RECIPE_GPU_SHORT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Structured harness outcome for worker / attestation surfaces."""

    ok: bool
    stage: str
    message: str
    gate: GateResult | None = None
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "stage": self.stage,
            "message": self.message,
            "mid_run_miner_mutation": False,
        }
        if self.gate is not None:
            payload["gate"] = self.gate.as_dict()
            payload["attestation"] = self.gate.attestation_metadata()
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _base_metadata() -> dict[str, Any]:
    pin = prod_data_window()
    loader = build_loader(pin)
    plan = loader.plan()
    return {
        "data_window": plan.as_dict(),
        "prod_token_budget_pin": 2_500_000_000,
        "token_budget_prod": 2_500_000_000,
        "equal_offset": pin.token_start_offset,
        "smoke_budget_env": "PRISM_RECIPE_TOKEN_BUDGET",
        "resolved_budget": resolve_token_budget(),
        "master_fineweb_mount_required": False,
        "workers_load_hf_themselves": True,
        "mid_run_miner_mutation": False,
        "image_contents": [
            "rules",
            "harness",
            "loader",
            "llm_gate",
            "tiny_1m",
        ],
    }


def preflight(*, skip_gate: bool | None = None) -> RunOutcome:
    """Validate env + data plan + LLM gate without starting train."""
    meta = _base_metadata()
    arch_src, train_src = read_sealed_sources()
    meta["sealed_sources"] = {
        "architecture_chars": len(arch_src),
        "training_chars": len(train_src),
    }

    if skip_gate is None:
        skip_gate = os.environ.get("PRISM_RECIPE_SMOKE_SKIP_GATE", "").strip() in {
            "1",
            "true",
            "yes",
        }

    if skip_gate:
        from datetime import datetime, timezone

        from prism_recipe.llm_gate import rules_digest

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
        return RunOutcome(
            ok=False,
            stage="llm_gate",
            message=gate.reason,
            gate=gate,
            metadata=meta,
        )
    return RunOutcome(
        ok=True,
        stage="preflight",
        message="preflight_ok",
        gate=gate,
        metadata=meta,
    )


def run_train(*, smoke: bool | None = None) -> RunOutcome:
    """Full train path.

    LLM rules gate **must** run first. When the gate returns ``ok:false``,
    train does not start and structured gate + attestation metadata are
    returned for the worker result / ExecutionProof path.

    Smoke mode (default when ``PRISM_RECIPE_SMOKE=1`` or tiny budget env is set
    without a full train deployment) runs offline tiny-1m steps on a mocked HF
    fixture under a small ``token_budget``.
    """
    if smoke is None:
        smoke = os.environ.get("PRISM_RECIPE_SMOKE", "").strip() in {"1", "true", "yes"}
        # Implicit smoke when callers only set a tiny budget override.
        if not smoke and os.environ.get("PRISM_RECIPE_TOKEN_BUDGET"):
            try:
                if resolve_token_budget() <= 100_000:
                    smoke = True
            except ValueError:
                smoke = False

    if smoke and not _want_gpu_short():
        result = run_smoke_train()
        meta = dict(result.metadata or {})
        meta.update(
            {
                "smoke": True,
                "param_count": result.param_count,
                "tokens_seen": result.tokens_seen,
                "steps": result.steps,
                "final_loss": result.final_loss,
                "architecture": result.architecture,
                "device": result.device,
                "token_budget": result.token_budget,
            }
        )
        return RunOutcome(
            ok=result.ok,
            stage=result.stage,
            message=result.message,
            gate=result.gate,
            metadata=meta,
        )

    # Real CUDA short train (~1M params, 50k–200k tokens) for live GPU score path.
    if _want_gpu_short():
        from prism_recipe.gpu_train import run_gpu_short_train

        result = run_gpu_short_train(require_cuda=True)
        meta = dict(result.metadata or {})
        meta.update(
            {
                "gpu_short": True,
                "smoke": False,
                "param_count": result.param_count,
                "tokens_consumed": result.tokens_consumed,
                "tokens_seen": result.tokens_consumed,
                "steps": result.steps,
                "final_loss": result.final_loss,
                "architecture": result.architecture,
                "device": result.device,
                "token_budget": result.token_budget,
                "cuda_name": result.cuda_name,
                "not_tinylm_fingerprint": True,
            }
        )
        if result.run_manifest is not None:
            meta["run_manifest"] = result.run_manifest
        return RunOutcome(
            ok=result.ok,
            stage=result.stage,
            message=result.message,
            gate=result.gate,
            metadata=meta,
        )

    # Non-smoke prod train still requires gate; full 2.5B loop is out of scope
    # for this milestone (pin only). Prefer smoke for local / CI proof.
    outcome = preflight(skip_gate=False)
    if not outcome.ok:
        return outcome
    return RunOutcome(
        ok=False,
        stage="train",
        message="prod_train_requires_deployed_worker",
        gate=outcome.gate,
        metadata={
            **(outcome.metadata or {}),
            "hint": (
                "set PRISM_RECIPE_SMOKE=1 for offline tiny-1m smoke, or "
                "PRISM_RECIPE_GPU_SHORT=1 for real CUDA short train"
            ),
        },
    )
