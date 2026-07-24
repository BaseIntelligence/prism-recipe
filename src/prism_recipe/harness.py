"""Recipe harness entrypoints (stubs).

Product model: architecture + training + train loop live **inside** this
image. Miners do not submit a separate architecture ZIP for this product path.
They supply ``OPENROUTER_API_KEY`` and Lium deploy environment only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prism_recipe.config import prod_data_window, resolve_token_budget
from prism_recipe.loader import build_loader
from prism_recipe.llm_gate import GateResult, run_rules_gate


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
        }
        if self.gate is not None:
            payload["gate"] = self.gate.as_dict()
            payload["attestation"] = self.gate.attestation_metadata()
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def preflight() -> RunOutcome:
    """Validate env + plans without starting train (stub)."""
    pin = prod_data_window()
    loader = build_loader(pin)
    plan = loader.plan()
    gate = run_rules_gate()
    meta = {
        "data_window": plan.as_dict(),
        "prod_token_budget_pin": 2_500_000_000,
        "token_budget_prod": 2_500_000_000,
        "equal_offset": pin.token_start_offset,
        "smoke_budget_env": "PRISM_RECIPE_TOKEN_BUDGET",
        "resolved_budget": resolve_token_budget(),
        "master_fineweb_mount_required": False,
        "workers_load_hf_themselves": True,
    }
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


def run_train() -> RunOutcome:
    """Full train path.

    LLM rules gate **must** run first via preflight. When the gate returns
    ``ok:false``, train does not start and the structured gate + attestation
    metadata are returned for the worker result / ExecutionProof path.
    """
    outcome = preflight()
    if not outcome.ok:
        # Gate (or earlier stage) blocked train — never start train on reject.
        return outcome
    return RunOutcome(
        ok=False,
        stage="train",
        message="train_not_implemented",
        gate=outcome.gate,
        metadata=outcome.metadata,
    )
