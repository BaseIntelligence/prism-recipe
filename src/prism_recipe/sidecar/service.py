"""Attestation sidecar service: measure, sign, answer, run loop."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prism_recipe.attestation.payload import (
    AttestationPayload,
    AttestationPayloadError,
    SignedAttestation,
    compute_build_secret_response,
    derive_attestation_key,
    sign_attestation_payload,
)
from prism_recipe.attestation_secret import (
    AttestationSecretError,
    read_attestation_hmac_key,
)
from prism_recipe.sidecar.config import SidecarConfig
from prism_recipe.sidecar.errors import SidecarError, SidecarReachabilityError
from prism_recipe.sidecar.measure import measure_and_compare
from prism_recipe.sidecar.schedule import next_interval_delay_seconds, plan_challenge_phases
from prism_recipe.sidecar.transport import ChallengeTransport
from prism_recipe.sidecar.types import Challenge, ChallengePhase
from prism_recipe.sidecar.wire import signed_attestation_to_wire, signed_from_wire


@dataclass(frozen=True, slots=True)
class ChallengeAnswer:
    """Local result of answering one challenge (signed + tamper evidence)."""

    signed: SignedAttestation
    phase: ChallengePhase
    baked_manifest_match: bool
    mismatched_paths: tuple[str, ...]


class AttestationSidecar:
    """In-image sidecar that answers BASE attestation challenges."""

    def __init__(self, config: SidecarConfig) -> None:
        self._config = config

    @property
    def config(self) -> SidecarConfig:
        return self._config

    def answer_challenge(self, challenge: Challenge) -> ChallengeAnswer:
        """Self-measure sealed surface, sign payload, return answer envelope."""
        build_secret = self._read_secret()
        live_hashes, match, mismatched = measure_and_compare(root=self._config.root)
        try:
            build_secret_response = compute_build_secret_response(
                build_secret=build_secret, nonce=challenge.nonce
            )
            payload = AttestationPayload(
                nonce=challenge.nonce,
                digest=self._config.digest,
                pod_id=self._config.pod_id,
                variant=self._config.variant,
                sealed_manifest_hashes=live_hashes,
                build_secret_response=build_secret_response,
            )
            signing_key = derive_attestation_key(build_secret)
            signed = sign_attestation_payload(payload, signing_key=signing_key)
        except AttestationPayloadError as exc:
            raise SidecarError(str(exc)) from exc
        return ChallengeAnswer(
            signed=signed,
            phase=challenge.phase,
            baked_manifest_match=match,
            mismatched_paths=mismatched,
        )

    def run(
        self,
        *,
        transport: ChallengeTransport,
        rng: random.Random | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> int:
        """Answer challenges for the planned phase schedule.

        Returns 0 on full success. Returns non-zero if BASE cannot be reached
        within ``retry_budget`` for any phase (fail closed).
        """
        rng = rng if rng is not None else random.Random()
        sleep = sleep_fn if sleep_fn is not None else time.sleep
        phases = plan_challenge_phases(interval_count=self._config.interval_count)
        for phase in phases:
            if phase is ChallengePhase.INTERVAL:
                delay = next_interval_delay_seconds(
                    rng=rng,
                    min_s=self._config.interval_min_s,
                    max_s=self._config.interval_max_s,
                )
                if delay > 0:
                    sleep(delay)
            try:
                self._answer_phase_with_retries(transport=transport, phase=phase)
            except SidecarReachabilityError:
                return 2
            except SidecarError:
                return 3
        return 0

    def _answer_phase_with_retries(
        self,
        *,
        transport: ChallengeTransport,
        phase: ChallengePhase,
    ) -> ChallengeAnswer:
        budget = self._config.retry_budget
        last_exc: SidecarReachabilityError | None = None
        for attempt in range(budget):
            try:
                challenge = transport.fetch_challenge(phase=phase)
                answer = self.answer_challenge(challenge)
                wire = signed_attestation_to_wire(answer.signed)
                wire["phase"] = phase.value
                if answer.mismatched_paths:
                    wire["baked_manifest_match"] = False
                    wire["mismatched_paths"] = list(answer.mismatched_paths)
                else:
                    wire["baked_manifest_match"] = True
                    wire["mismatched_paths"] = []
                transport.submit_answer(wire)
                return answer
            except SidecarReachabilityError as exc:
                last_exc = exc
                if attempt + 1 >= budget:
                    break
                continue
        assert last_exc is not None
        raise last_exc

    def _read_secret(self) -> bytes:
        try:
            return read_attestation_hmac_key(self._config.secret_path)
        except AttestationSecretError as exc:
            raise SidecarError(str(exc)) from exc

    @staticmethod
    def signed_from_wire(wire: dict[str, Any]) -> SignedAttestation:
        """Rehydrate wire JSON (tests / local verify)."""
        return signed_from_wire(wire)
