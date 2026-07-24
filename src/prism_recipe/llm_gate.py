"""Pre-train OpenRouter LLM rules gate (stub).

Miner supplies OPENROUTER_API_KEY. Rules are loaded from image-bundled
``.rules/`` and must not be mutable mid-run. Full client + attestation
metadata land in a later feature.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from prism_recipe.config import DEFAULT_OPENROUTER_MODEL, OPENROUTER_API_KEY_ENV

# Default location of bundled rules inside the recipe image / checkout.
DEFAULT_RULES_DIR = Path(__file__).resolve().parents[2] / ".rules"


@dataclass(frozen=True, slots=True)
class GateResult:
    """Structured gate outcome (never a raw stack for expected rejects)."""

    ok: bool
    reason: str
    rules_digest: str
    model: str
    checked_at: str
    decision: str  # "pass" | "fail" | "error"

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "rules_digest": self.rules_digest,
            "model": self.model,
            "checked_at": self.checked_at,
            "decision": self.decision,
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


def run_rules_gate(
    *,
    architecture_source: str = "",
    training_source: str = "",
    rules_dir: Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> GateResult:
    """Run the pre-train rules gate (stub: key check + rules digest only).

    Full OpenRouter call is implemented in prism-recipe-llm-gate-attested.
    Missing key yields structured fail (no stack crash).
    """
    del architecture_source, training_source  # reserved for full gate
    checked_at = datetime.now(timezone.utc).isoformat()
    digest = rules_digest(rules_dir)
    resolved_model = model or DEFAULT_OPENROUTER_MODEL
    key = api_key if api_key is not None else os.environ.get(OPENROUTER_API_KEY_ENV, "")
    if not key or not str(key).strip():
        return GateResult(
            ok=False,
            reason="missing_openrouter_api_key",
            rules_digest=digest,
            model=resolved_model,
            checked_at=checked_at,
            decision="fail",
        )
    return GateResult(
        ok=False,
        reason="llm_gate_not_implemented",
        rules_digest=digest,
        model=resolved_model,
        checked_at=checked_at,
        decision="error",
    )
