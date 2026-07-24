"""Scaffold smoke tests (no network, no secrets)."""

from __future__ import annotations

import os
from pathlib import Path

from prism_recipe import __version__
from prism_recipe.config import PROD_TOKEN_BUDGET, prod_data_window, resolve_token_budget
from prism_recipe.llm_gate import rules_digest, run_rules_gate
from prism_recipe.loader import build_loader


def test_version_semver_shape() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_prod_token_budget_pin() -> None:
    pin = prod_data_window()
    assert pin.token_budget == 2_500_000_000
    assert pin.token_budget == PROD_TOKEN_BUDGET
    assert pin.epochs == 1
    assert pin.token_start_offset == 0


def test_smoke_budget_env_override(monkeypatch: object) -> None:
    monkeypatch.setenv("PRISM_RECIPE_TOKEN_BUDGET", "1000000")  # type: ignore[attr-defined]
    assert resolve_token_budget() == 1_000_000
    plan = build_loader().plan()
    assert plan.token_budget == 1_000_000
    assert plan.pin.token_start_offset == 0  # identity unchanged


def test_loader_stub_raises() -> None:
    loader = build_loader()
    try:
        next(loader.iter_tokens())
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def test_rules_digest_stable() -> None:
    d1 = rules_digest()
    d2 = rules_digest()
    assert len(d1) == 64
    assert d1 == d2
    # Bundled rules should exist in the checkout / image.
    rules_root = Path(__file__).resolve().parents[1] / ".rules"
    assert rules_root.is_dir()
    assert any(rules_root.rglob("*.md"))


def test_missing_openrouter_key_is_structured_fail(monkeypatch: object) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # type: ignore[attr-defined]
    # Ensure empty even if host exports a key (tests must not print it).
    os.environ.pop("OPENROUTER_API_KEY", None)
    result = run_rules_gate(api_key="")
    assert result.ok is False
    assert result.reason == "missing_openrouter_api_key"
    assert result.decision == "fail"
    meta = result.attestation_metadata()
    assert "llm_gate" in meta
    assert meta["llm_gate"]["rules_digest"] == result.rules_digest
