"""Challenge transport: Protocol + fake + optional httpx client."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from prism_recipe.sidecar.errors import SidecarReachabilityError
from prism_recipe.sidecar.types import Challenge, ChallengePhase, parse_challenge


@runtime_checkable
class ChallengeTransport(Protocol):
    """Mockable BASE challenge channel."""

    def fetch_challenge(self, *, phase: ChallengePhase) -> Challenge:
        """Pull the next challenge for ``phase`` from BASE."""
        ...

    def submit_answer(self, answer: Mapping[str, Any]) -> None:
        """POST the signed wire answer to BASE."""
        ...


@dataclass
class FakeChallengeTransport:
    """In-memory transport for unit tests (no network)."""

    challenges_by_phase: MutableMapping[ChallengePhase, Challenge]
    submitted: list[dict[str, Any]] = field(default_factory=list)
    fetch_failures_remaining: int = 0
    submit_failures_remaining: int = 0

    def fetch_challenge(self, *, phase: ChallengePhase) -> Challenge:
        if self.fetch_failures_remaining > 0:
            self.fetch_failures_remaining -= 1
            raise SidecarReachabilityError("fake fetch failure")
        challenge = self.challenges_by_phase.get(phase)
        if challenge is None:
            # Allow a single INTERVAL challenge to be reused for multiple intervals.
            if phase is ChallengePhase.INTERVAL:
                challenge = self.challenges_by_phase.get(ChallengePhase.INTERVAL)
            if challenge is None:
                raise SidecarReachabilityError(f"no fake challenge for phase={phase}")
        return challenge

    def submit_answer(self, answer: Mapping[str, Any]) -> None:
        if self.submit_failures_remaining > 0:
            self.submit_failures_remaining -= 1
            raise SidecarReachabilityError("fake submit failure")
        self.submitted.append(dict(answer))


class HttpxChallengeTransport:
    """Optional HTTP transport against BASE (requires httpx at call time).

    Endpoints (convention, BASE may refine later):
    * GET  ``{base_url}/v1/attestation/challenge?phase={phase}``
    * POST ``{base_url}/v1/attestation/answer``
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 10.0,
        client: Any | None = None,
    ) -> None:
        if not base_url.strip():
            raise SidecarReachabilityError("base_url must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client

    def fetch_challenge(self, *, phase: ChallengePhase) -> Challenge:
        client = self._ensure_client()
        url = f"{self._base_url}/v1/attestation/challenge"
        try:
            response = client.get(
                url, params={"phase": phase.value}, timeout=self._timeout_s
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise SidecarReachabilityError(f"fetch_challenge failed: {exc}") from exc
        if not isinstance(data, dict):
            raise SidecarReachabilityError("challenge response must be a JSON object")
        try:
            return parse_challenge(data)
        except ValueError as exc:
            raise SidecarReachabilityError(str(exc)) from exc

    def submit_answer(self, answer: Mapping[str, Any]) -> None:
        client = self._ensure_client()
        url = f"{self._base_url}/v1/attestation/answer"
        try:
            response = client.post(url, json=dict(answer), timeout=self._timeout_s)
            response.raise_for_status()
        except Exception as exc:
            raise SidecarReachabilityError(f"submit_answer failed: {exc}") from exc

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:
            raise SidecarReachabilityError(
                "httpx is required for HttpxChallengeTransport"
            ) from exc
        self._client = httpx.Client()
        return self._client
