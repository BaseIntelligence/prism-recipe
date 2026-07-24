"""Unit tests for pre-train OpenRouter LLM rules gate (mocked HTTP)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from prism_recipe.config import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_CHAT_URL,
    OPENROUTER_PROMPT_VERSION,
)
from prism_recipe.harness import preflight, run_train
from prism_recipe.llm_gate import (
    GateResult,
    build_gate_messages,
    prompt_hash_for_messages,
    rules_digest,
    run_rules_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = REPO_ROOT / ".rules"


def test_default_openrouter_model_is_grok_45() -> None:
    """VAL-RECIPE-SYS-001: evaluation/rules-gate default is x-ai/grok-4.5."""
    assert DEFAULT_OPENROUTER_MODEL == "x-ai/grok-4.5"


def _or_response(
    *,
    status_code: int = 200,
    content: str | dict[str, Any] | None = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    if body is not None:
        payload = body
    else:
        if isinstance(content, dict):
            message_content = json.dumps(content)
        elif content is None:
            message_content = json.dumps({"ok": True, "reason": "pass", "reason_codes": []})
        else:
            message_content = content
        payload = {
            "id": "gen-test",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message_content},
                    "finish_reason": "stop",
                }
            ],
        }
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", OPENROUTER_CHAT_URL),
    )


def _mock_post_factory(response: httpx.Response | Exception):
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        captured["json"] = kwargs.get("json") or {}
        if isinstance(response, Exception):
            raise response
        return response

    return _post, captured


def test_missing_key_structured_fail_no_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    os.environ.pop("OPENROUTER_API_KEY", None)

    called = {"n": 0}

    def boom(*_a: Any, **_k: Any) -> httpx.Response:
        called["n"] += 1
        raise AssertionError("http must not be called when key missing")

    result = run_rules_gate(api_key="", http_post=boom)
    assert result.ok is False
    assert result.reason == "missing_openrouter_api_key"
    assert result.decision == "fail"
    assert result.rules_digest == rules_digest(RULES_DIR)
    assert result.model == DEFAULT_OPENROUTER_MODEL
    assert result.checked_at
    assert called["n"] == 0
    # Structured surfaces only — no stack / key material.
    d = result.as_dict()
    assert set(d) >= {"ok", "reason", "rules_digest", "model", "checked_at", "decision"}
    blob = json.dumps(d)
    assert "sk-" not in blob
    assert "Traceback" not in blob


def test_gate_pass_attestation_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    response = _or_response(content={"ok": True, "reason": "compliant", "reason_codes": []})
    post, captured = _mock_post_factory(response)

    result = run_rules_gate(
        architecture_source="def build_model():\n    return Tiny()\n",
        training_source="def train():\n    pass\n",
        api_key="test-key-not-real",
        http_post=post,
        rules_dir=RULES_DIR,
    )
    assert result.ok is True
    assert result.decision == "pass"
    assert result.reason == "compliant"
    assert len(result.rules_digest) == 64
    assert len(result.prompt_hash) == 64
    assert result.model == DEFAULT_OPENROUTER_MODEL
    assert result.checked_at

    meta = result.attestation_metadata()
    gate_meta = meta["llm_gate"]
    assert gate_meta["rules_digest"] == result.rules_digest
    assert gate_meta["model"] == result.model
    assert gate_meta["decision"] == "pass"
    assert gate_meta["ok"] is True
    assert gate_meta["checked_at"] == result.checked_at
    assert gate_meta["prompt_hash"] == result.prompt_hash
    assert gate_meta["prompt_version"] == OPENROUTER_PROMPT_VERSION

    # Request used pinned model + Bearer header (tests must not print key).
    assert captured["url"] == OPENROUTER_CHAT_URL
    assert captured["json"]["model"] == DEFAULT_OPENROUTER_MODEL
    assert captured["json"]["stream"] is False
    assert "Authorization" in captured["headers"]
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    # Never embed key in attestation JSON.
    assert "test-key-not-real" not in json.dumps(meta)


def test_gate_reject_blocks_with_structured_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    response = _or_response(
        content={
            "ok": False,
            "reason": "pretrained_weights",
            "reason_codes": ["pretrained_weights"],
        }
    )
    post, _ = _mock_post_factory(response)
    result = run_rules_gate(
        architecture_source="from_pretrained('gpt2')",
        training_source="load_checkpoint('/weights.bin')",
        api_key="test-key-not-real",
        http_post=post,
    )
    assert result.ok is False
    assert result.decision == "fail"
    assert result.reason == "pretrained_weights"
    assert "pretrained_weights" in result.reason_codes
    assert result.prompt_hash
    assert result.rules_digest


def test_provider_http_error_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    post, _ = _mock_post_factory(_or_response(status_code=503, content="upstream"))
    result = run_rules_gate(api_key="test-key-not-real", http_post=post)
    assert result.ok is False
    assert result.decision == "error"
    assert result.reason == "openrouter_http_error"
    assert result.prompt_hash


def test_auth_failed_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    post, _ = _mock_post_factory(_or_response(status_code=401, content="nope"))
    result = run_rules_gate(api_key="bad-key", http_post=post)
    assert result.ok is False
    assert result.reason == "openrouter_auth_failed"
    assert result.decision == "error"


def test_timeout_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    post, _ = _mock_post_factory(httpx.ReadTimeout("slow"))
    result = run_rules_gate(api_key="test-key-not-real", http_post=post)
    assert result.ok is False
    assert result.reason == "openrouter_timeout"
    assert result.decision == "error"
    assert result.prompt_hash


def test_malformed_model_json_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    post, _ = _mock_post_factory(_or_response(content="not-json-at-all"))
    result = run_rules_gate(api_key="test-key-not-real", http_post=post)
    assert result.ok is False
    assert result.reason == "model_response_malformed"
    assert result.decision == "error"


def test_prompt_hash_stable_for_same_inputs() -> None:
    digest = rules_digest(RULES_DIR)
    from prism_recipe.llm_gate import load_rules_text

    rules = load_rules_text(RULES_DIR)
    m1 = build_gate_messages(
        rules_text=rules,
        architecture_source="a",
        training_source="b",
        rules_digest_value=digest,
    )
    m2 = build_gate_messages(
        rules_text=rules,
        architecture_source="a",
        training_source="b",
        rules_digest_value=digest,
    )
    h1 = prompt_hash_for_messages(m1, model=DEFAULT_OPENROUTER_MODEL)
    h2 = prompt_hash_for_messages(m2, model=DEFAULT_OPENROUTER_MODEL)
    assert h1 == h2
    assert len(h1) == 64
    # Different sources → different hash.
    m3 = build_gate_messages(
        rules_text=rules,
        architecture_source="different",
        training_source="b",
        rules_digest_value=digest,
    )
    assert prompt_hash_for_messages(m3, model=DEFAULT_OPENROUTER_MODEL) != h1


def test_run_train_blocks_when_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate MUST run before train; ok:false blocks train stage."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    os.environ.pop("OPENROUTER_API_KEY", None)
    outcome = run_train()
    assert outcome.ok is False
    assert outcome.stage == "llm_gate"
    assert outcome.gate is not None
    assert outcome.gate.ok is False
    assert outcome.gate.reason == "missing_openrouter_api_key"
    payload = outcome.as_dict()
    assert payload["gate"]["ok"] is False
    assert "llm_gate" in payload["attestation"]
    assert payload["attestation"]["llm_gate"]["rules_digest"]
    assert payload["attestation"]["llm_gate"]["decision"] == "fail"
    assert payload["attestation"]["llm_gate"]["checked_at"]
    assert "message" in payload
    assert "Traceback" not in json.dumps(payload)


def test_preflight_pass_includes_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    response = _or_response(content={"ok": True, "reason": "pass", "reason_codes": []})
    post, _ = _mock_post_factory(response)

    # Inject mock via run_rules_gate monkeypatch so harness path is exercised.
    import prism_recipe.harness as harness_mod

    def fake_gate(**kwargs: Any) -> GateResult:
        return run_rules_gate(http_post=post, api_key="test-key-not-real", **kwargs)

    monkeypatch.setattr(harness_mod, "run_rules_gate", fake_gate)
    outcome = preflight()
    assert outcome.ok is True
    assert outcome.stage == "preflight"
    assert outcome.gate is not None
    assert outcome.gate.ok is True
    att = outcome.as_dict()["attestation"]["llm_gate"]
    assert att["decision"] == "pass"
    assert att["prompt_hash"]
    assert att["rules_digest"]
    assert att["model"]
    assert att["checked_at"]


def test_fenced_json_content_accepted() -> None:
    fenced = '```json\n{"ok": false, "reason": "gate_bypass", "reason_codes": ["gate_bypass"]}\n```'
    post, _ = _mock_post_factory(_or_response(content=fenced))
    result = run_rules_gate(api_key="k", http_post=post)
    assert result.ok is False
    assert result.reason == "gate_bypass"
    assert result.decision == "fail"
