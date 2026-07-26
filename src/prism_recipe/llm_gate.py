"""Pre-train OpenRouter LLM rules gate (VAL-RECIPE-003 / VAL-RECIPE-004).

Miner supplies ``OPENROUTER_API_KEY`` in the worker env. The gate:

1. Loads image-bundled ``.rules/*.md`` (immutable in the image).
2. Calls OpenRouter with a pinned model.
3. Always returns structured JSON on the success path of the HTTP client
   (provider HTTP 200 treated as transport success; recipe surfaces
   ``{ok, reason, rules_digest, model, ...}`` — never raw stack traces for
   expected rejects).
4. Missing key → structured fail (no crash).
5. Attaches attestation metadata for ExecutionProof / worker result:
   ``rules_digest``, ``model``, ``decision``, ``checked_at``, ``prompt_hash``.

Gate MUST run before train (see ``harness.run_train`` / ``preflight``).
Never print API keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from prism_recipe.config import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_URL,
    OPENROUTER_HTTP_TIMEOUT_SECONDS,
    OPENROUTER_MAX_SOURCE_CHARS,
    OPENROUTER_PROMPT_VERSION,
)

logger = logging.getLogger(__name__)

# Default location of bundled rules inside the recipe image / checkout.
DEFAULT_RULES_DIR = Path(__file__).resolve().parents[2] / ".rules"

# HTTP transport injectable for unit tests (returns httpx.Response-like).
HttpPoster = Callable[..., httpx.Response]

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Structured gate outcome (never a raw stack for expected rejects)."""

    ok: bool
    reason: str
    rules_digest: str
    model: str
    checked_at: str
    decision: str  # "pass" | "fail" | "error"
    prompt_hash: str = ""
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "rules_digest": self.rules_digest,
            "model": self.model,
            "checked_at": self.checked_at,
            "decision": self.decision,
            "prompt_hash": self.prompt_hash,
            "reason_codes": list(self.reason_codes),
        }

    def attestation_metadata(self) -> dict[str, object]:
        """Fields intended for ExecutionProof / worker result metadata."""
        return {
            "llm_gate": {
                "rules_digest": self.rules_digest,
                "model": self.model,
                "decision": self.decision,
                "ok": self.ok,
                "checked_at": self.checked_at,
                "reason": self.reason,
                "prompt_hash": self.prompt_hash,
                "prompt_version": OPENROUTER_PROMPT_VERSION,
                "reason_codes": list(self.reason_codes),
            }
        }


def rules_digest(rules_dir: Path | None = None) -> str:
    """SHA-256 over sorted rule file contents (stable across runs)."""
    root = rules_dir or DEFAULT_RULES_DIR
    if not root.is_dir():
        return hashlib.sha256(b"").hexdigest()
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix().encode()
        h.update(rel)
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def load_rules_text(rules_dir: Path | None = None) -> str:
    """Concatenate image-bundled rules for the gate prompt (read-only)."""
    root = rules_dir or DEFAULT_RULES_DIR
    if not root.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(root.rglob("*.md")):
        parts.append(f"# {path.relative_to(root).as_posix()}\n\n{path.read_text()}")
    return "\n\n---\n\n".join(parts)


def _clip(text: str, *, max_chars: int) -> str:
    if max_chars < 1 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n/* truncated at {max_chars} chars */\n"


def build_gate_messages(
    *,
    rules_text: str,
    architecture_source: str,
    training_source: str,
    rules_digest_value: str,
    max_source_chars: int = OPENROUTER_MAX_SOURCE_CHARS,
) -> list[dict[str, str]]:
    """Build system+user messages for the pinned OpenRouter call."""
    system = (
        "You are the PRISM recipe pre-train rules gate. Judge architecture and "
        "training source against the provided image-bundled rules only. "
        "Respond with a single JSON object (no markdown fences) with keys: "
        'ok (boolean), reason (short snake_case string), reason_codes (array of '
        "short snake_case strings). "
        "ok=true only when sources clearly comply. ok=false for any clear "
        "violation or when compliance cannot be established. Never invent "
        "file contents beyond what is provided."
    )
    user_payload = {
        "prompt_version": OPENROUTER_PROMPT_VERSION,
        "rules_digest": rules_digest_value,
        "rules": rules_text,
        "architecture_source": _clip(architecture_source, max_chars=max_source_chars),
        "training_source": _clip(training_source, max_chars=max_source_chars),
        "instructions": [
            "Evaluate against training, anti-cheat, security, and acceptance rules.",
            "Reject pretrained weight loads, multi-epoch window rescans, gate bypass, "
            "rules mutation, hardcoded scores, and secret leakage patterns.",
            "If sources are empty, set ok=false with reason empty_sources.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":")),
        },
    ]


def prompt_hash_for_messages(messages: list[dict[str, str]], *, model: str) -> str:
    """Stable SHA-256 over the exact model pin + messages sent to OpenRouter."""
    canonical = json.dumps(
        {"model": model, "messages": messages, "prompt_version": OPENROUTER_PROMPT_VERSION},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _result(
    *,
    ok: bool,
    reason: str,
    rules_digest_value: str,
    model: str,
    checked_at: str,
    decision: str,
    prompt_hash: str = "",
    reason_codes: tuple[str, ...] | list[str] = (),
) -> GateResult:
    codes = tuple(str(c) for c in reason_codes if str(c).strip())
    return GateResult(
        ok=ok,
        reason=reason,
        rules_digest=rules_digest_value,
        model=model,
        checked_at=checked_at,
        decision=decision,
        prompt_hash=prompt_hash,
        reason_codes=codes,
    )


def _parse_model_json(content: str) -> dict[str, Any] | None:
    """Parse model content into a dict; tolerate light fence / prose wrapping."""
    text = (content or "").strip()
    if not text:
        return None
    # Strip common markdown fences.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _decision_from_payload(payload: Mapping[str, Any]) -> tuple[bool, str, tuple[str, ...]]:
    """Map model JSON to ok/reason/reason_codes (fail closed on ambiguity)."""
    raw_codes = payload.get("reason_codes") or payload.get("codes") or []
    codes: list[str] = []
    if isinstance(raw_codes, list):
        codes = [str(c).strip() for c in raw_codes if str(c).strip()]
    elif isinstance(raw_codes, str) and raw_codes.strip():
        codes = [raw_codes.strip()]

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "model_reject" if payload.get("ok") is False else "model_unparseable_reason"
    else:
        reason = reason.strip()

    ok_raw = payload.get("ok")
    if isinstance(ok_raw, bool):
        return ok_raw, reason, tuple(codes)
    if isinstance(ok_raw, str):
        lowered = ok_raw.strip().lower()
        if lowered in {"true", "pass", "ok", "yes", "1"}:
            return True, reason if reason != "model_unparseable_reason" else "pass", tuple(codes)
        if lowered in {"false", "fail", "reject", "no", "0"}:
            return False, reason if reason != "model_unparseable_reason" else "model_reject", tuple(
                codes
            )
    # Missing / wrong type → fail closed (structured).
    return False, "model_response_malformed", tuple(codes) or ("model_response_malformed",)


def _post_openrouter(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    http_post: HttpPoster | None,
    timeout: float,
) -> httpx.Response:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Optional product attribution headers (no secrets).
        "HTTP-Referer": "https://github.com/BaseIntelligence/prism-recipe",
        "X-Title": "prism-recipe-llm-gate",
    }
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if http_post is not None:
        return http_post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json=body,
            timeout=timeout,
        )
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        return client.post(OPENROUTER_CHAT_URL, headers=headers, json=body)


def _extract_assistant_content(response_json: Mapping[str, Any]) -> str | None:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    # Some providers return list-of-parts content.
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts) if parts else None
    return None


def run_rules_gate(
    *,
    architecture_source: str = "",
    training_source: str = "",
    rules_dir: Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    http_post: HttpPoster | None = None,
    timeout: float | None = None,
    max_source_chars: int = OPENROUTER_MAX_SOURCE_CHARS,
) -> GateResult:
    """Run the pre-train OpenRouter rules gate.

    Always returns a :class:`GateResult` for expected reject / infra paths.
    Does not raise for missing key, HTTP errors, or model reject payloads.
    Never logs or includes the API key in the result.
    """
    checked_at = _utc_now_iso()
    digest = rules_digest(rules_dir)
    resolved_model = model or DEFAULT_OPENROUTER_MODEL
    key = api_key if api_key is not None else os.environ.get(OPENROUTER_API_KEY_ENV, "")
    key = str(key).strip() if key is not None else ""

    if not key:
        return _result(
            ok=False,
            reason="missing_openrouter_api_key",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="fail",
            prompt_hash="",
            reason_codes=("missing_openrouter_api_key",),
        )

    rules_text = load_rules_text(rules_dir)
    if not rules_text.strip():
        return _result(
            ok=False,
            reason="rules_missing",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash="",
            reason_codes=("rules_missing",),
        )

    messages = build_gate_messages(
        rules_text=rules_text,
        architecture_source=architecture_source or "",
        training_source=training_source or "",
        rules_digest_value=digest,
        max_source_chars=max_source_chars,
    )
    p_hash = prompt_hash_for_messages(messages, model=resolved_model)
    resolved_timeout = (
        float(timeout) if timeout is not None else float(OPENROUTER_HTTP_TIMEOUT_SECONDS)
    )

    try:
        response = _post_openrouter(
            api_key=key,
            model=resolved_model,
            messages=messages,
            http_post=http_post,
            timeout=resolved_timeout,
        )
    except httpx.TimeoutException:
        logger.warning("openrouter gate timeout model=%s", resolved_model)
        return _result(
            ok=False,
            reason="openrouter_timeout",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_timeout",),
        )
    except httpx.HTTPError as exc:
        # Never include request headers / key material.
        logger.warning("openrouter gate http error type=%s", type(exc).__name__)
        return _result(
            ok=False,
            reason="openrouter_unavailable",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_unavailable",),
        )
    except Exception as exc:  # noqa: BLE001 — gate must never crash train path
        logger.warning("openrouter gate unexpected error type=%s", type(exc).__name__)
        return _result(
            ok=False,
            reason="openrouter_error",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_error",),
        )

    status = int(getattr(response, "status_code", 0) or 0)
    if status in {401, 403}:
        return _result(
            ok=False,
            reason="openrouter_auth_failed",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_auth_failed",),
        )
    if status == 429:
        return _result(
            ok=False,
            reason="openrouter_rate_limited",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_rate_limited",),
        )
    if status != 200:
        return _result(
            ok=False,
            reason="openrouter_http_error",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_http_error", f"http_{status}"),
        )

    try:
        response_json = response.json()
    except Exception:  # noqa: BLE001
        return _result(
            ok=False,
            reason="openrouter_body_not_json",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_body_not_json",),
        )
    if not isinstance(response_json, Mapping):
        return _result(
            ok=False,
            reason="openrouter_body_not_json",
            rules_digest_value=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("openrouter_body_not_json",),
        )

    # Prefer provider-reported model when present (still pin request model).
    returned_model = response_json.get("model")
    attested_model = (
        str(returned_model).strip()
        if isinstance(returned_model, str) and returned_model
        else resolved_model
    )

    content = _extract_assistant_content(response_json)
    if content is None:
        return _result(
            ok=False,
            reason="model_response_empty",
            rules_digest_value=digest,
            model=attested_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("model_response_empty",),
        )

    payload = _parse_model_json(content)
    if payload is None:
        return _result(
            ok=False,
            reason="model_response_malformed",
            rules_digest_value=digest,
            model=attested_model,
            checked_at=checked_at,
            decision="error",
            prompt_hash=p_hash,
            reason_codes=("model_response_malformed",),
        )

    ok, reason, codes = _decision_from_payload(payload)
    decision = "pass" if ok else "fail"
    return _result(
        ok=ok,
        reason=reason,
        rules_digest_value=digest,
        model=attested_model,
        checked_at=checked_at,
        decision=decision,
        prompt_hash=p_hash,
        reason_codes=codes or ((reason,) if reason else ()),
    )
