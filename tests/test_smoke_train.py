"""Tiny-1m smoke train + image contract tests (VAL-RECIPE-005 / VAL-RECIPE-006)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from prism_recipe.arch.tiny_1m import MAX_PARAMS, MODEL_VOCAB_SIZE, build_tiny_1m, count_parameters
from prism_recipe.cli import main as cli_main
from prism_recipe.harness import preflight, run_train
from prism_recipe.smoke_train import (
    ARCH_NAME,
    fixture_encode,
    run_smoke_train,
    tiny_fixture_documents,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_1m_param_count_under_cap() -> None:
    torch = pytest.importorskip("torch")
    model = build_tiny_1m(seed=0)
    n = count_parameters(model)
    assert n <= MAX_PARAMS
    assert n <= 1_500_000
    # Weight-tied ~1.05M range (platform tiny-1m parity).
    assert 500_000 <= n <= 1_500_000
    assert model.vocab_size == MODEL_VOCAB_SIZE
    del torch


def test_tiny_fixture_and_encode_stable() -> None:
    docs = tiny_fixture_documents(n_docs=4, words_per_doc=10)
    assert len(docs) == 4
    ids_a = fixture_encode(docs[0]["text"])
    ids_b = fixture_encode(docs[0]["text"])
    assert ids_a == ids_b
    assert all(0 < t < MODEL_VOCAB_SIZE for t in ids_a)


def test_smoke_train_offline_e2e() -> None:
    """End-to-end offline: fixture HF stream + tiny-1m under small token_budget."""
    pytest.importorskip("torch")
    result = run_smoke_train(
        token_budget=128,
        skip_gate=True,
        max_steps=2,
        seq_len=16,
        batch_size=2,
    )
    assert result.ok is True, result.as_dict()
    assert result.stage == "smoke_train"
    assert result.architecture == ARCH_NAME
    assert result.param_count <= MAX_PARAMS
    assert result.param_count > 0
    assert result.steps >= 1
    assert result.tokens_seen <= 128
    assert result.final_loss is not None
    assert result.final_loss == result.final_loss  # not NaN
    assert result.gate is not None
    assert result.gate.ok is True
    meta = result.metadata or {}
    assert meta.get("mid_run_miner_mutation") is False
    assert "rules" in meta.get("image_contents", [])
    assert "harness" in meta.get("image_contents", [])


def test_smoke_train_respects_gate_reject() -> None:
    from datetime import datetime, timezone

    from prism_recipe.llm_gate import GateResult, rules_digest

    bad = GateResult(
        ok=False,
        reason="unit_test_reject",
        rules_digest=rules_digest(),
        model="test",
        checked_at=datetime.now(timezone.utc).isoformat(),
        decision="fail",
        reason_codes=("unit_test_reject",),
    )
    result = run_smoke_train(token_budget=64, gate=bad)
    assert result.ok is False
    assert result.stage == "llm_gate"
    assert result.message == "unit_test_reject"
    assert result.steps == 0


def test_smoke_train_budget_stop_on_loader() -> None:
    pytest.importorskip("torch")
    # Tiny budget, large docs — tokens_seen must not exceed budget.
    docs = tiny_fixture_documents(n_docs=20, words_per_doc=80)
    result = run_smoke_train(
        token_budget=40,
        document_source=docs,
        skip_gate=True,
        max_steps=1,
        seq_len=8,
        batch_size=1,
    )
    assert result.ok is True
    assert result.tokens_seen <= 40


def test_harness_run_train_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("PRISM_RECIPE_SMOKE", "1")
    monkeypatch.setenv("PRISM_RECIPE_SMOKE_SKIP_GATE", "1")
    monkeypatch.setenv("PRISM_RECIPE_TOKEN_BUDGET", "128")
    outcome = run_train(smoke=True)
    assert outcome.ok is True, outcome.as_dict()
    assert outcome.metadata is not None
    assert outcome.metadata.get("smoke") is True
    assert outcome.metadata.get("mid_run_miner_mutation") is False
    assert outcome.gate is not None


def test_cli_smoke_skip_gate(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    os.environ.pop("OPENROUTER_API_KEY", None)
    code = cli_main(["smoke", "--skip-gate"])
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["mid_run_miner_mutation"] is False
    assert payload["stage"] == "smoke_train"


def test_dockerfile_bundles_sealed_contents() -> None:
    """Image recipe files declare sealed contents (VAL-RECIPE-005 contract)."""
    cpu = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    cuda = (REPO_ROOT / "Dockerfile.cuda").read_text(encoding="utf-8")
    for text in (cpu, cuda):
        assert "COPY .rules" in text
        assert "src/prism_recipe/harness.py" in text or "COPY src" in text
        assert "mid_run_miner_mutation" in text
        assert "tiny" in text.lower()
    assert "pytorch/pytorch" in cuda or "cuda" in cuda.lower()
    assert "cpu" in cpu.lower()


def test_miner_docs_list_deploy_env() -> None:
    miner = (REPO_ROOT / "docs" / "miner" / "README.md").read_text(encoding="utf-8")
    for needle in (
        "OPENROUTER_API_KEY",
        "LIUM_API_KEY",
        "chain.joinbase.ai",
        "binding",
        "digest",
    ):
        assert needle in miner, f"missing {needle} in miner docs"


def test_preflight_skip_gate_smoke_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_RECIPE_SMOKE_SKIP_GATE", "1")
    outcome = preflight()
    assert outcome.ok is True
    assert outcome.gate is not None
    assert outcome.metadata is not None
    assert outcome.metadata["mid_run_miner_mutation"] is False
    assert "tiny_1m" in outcome.metadata["image_contents"]
