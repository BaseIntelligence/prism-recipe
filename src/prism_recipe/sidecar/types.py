"""Challenge phase + challenge message types (parse at boundary)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never


class ChallengePhase(StrEnum):
    """When in the run the challenge is answered."""

    START = "start"
    INTERVAL = "interval"
    END = "end"


@dataclass(frozen=True, slots=True)
class Challenge:
    """BASE-issued challenge. Nonce is the single-use id from the nonce service."""

    nonce: str
    phase: ChallengePhase
    challenge_id: str | None = None

    def __post_init__(self) -> None:
        nonce = self.nonce.strip() if isinstance(self.nonce, str) else ""
        if not nonce:
            msg = "challenge.nonce must be non-empty"
            raise ValueError(msg)
        object.__setattr__(self, "nonce", nonce)
        if not isinstance(self.phase, ChallengePhase):
            object.__setattr__(self, "phase", ChallengePhase(str(self.phase)))


def parse_challenge(raw: dict[str, object]) -> Challenge:
    """Parse untrusted BASE JSON into a ``Challenge`` (boundary)."""
    nonce = raw.get("nonce")
    if not isinstance(nonce, str) or not nonce.strip():
        msg = "challenge JSON missing non-empty nonce"
        raise ValueError(msg)
    phase_raw = raw.get("phase", ChallengePhase.INTERVAL.value)
    if not isinstance(phase_raw, str):
        msg = "challenge.phase must be a string"
        raise ValueError(msg)
    phase_key = phase_raw.strip().lower()
    match phase_key:
        case "start":
            phase = ChallengePhase.START
        case "interval" | "random" | "mid":
            phase = ChallengePhase.INTERVAL
        case "end":
            phase = ChallengePhase.END
        case unreachable:
            msg = f"unknown challenge phase: {unreachable!r}"
            raise ValueError(msg)
    cid = raw.get("challenge_id")
    challenge_id = cid.strip() if isinstance(cid, str) and cid.strip() else None
    return Challenge(nonce=nonce, phase=phase, challenge_id=challenge_id)


def phase_label(phase: ChallengePhase) -> str:
    """Stable machine token for logs/evidence (not prose)."""
    match phase:
        case ChallengePhase.START:
            return "start"
        case ChallengePhase.INTERVAL:
            return "interval"
        case ChallengePhase.END:
            return "end"
        case unreachable:
            assert_never(unreachable)
