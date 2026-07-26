"""Pure challenge schedule: start → N random intervals → end."""

from __future__ import annotations

import random
from collections.abc import Sequence

from prism_recipe.sidecar.types import ChallengePhase


def plan_challenge_phases(*, interval_count: int) -> tuple[ChallengePhase, ...]:
    """Return the ordered phase list for one sidecar run.

    Always starts with START and ends with END. ``interval_count`` mid-run
    INTERVAL phases sit between them (may be zero).
    """
    if interval_count < 0:
        msg = "interval_count must be >= 0"
        raise ValueError(msg)
    mids = (ChallengePhase.INTERVAL,) * interval_count
    return (ChallengePhase.START, *mids, ChallengePhase.END)


def next_interval_delay_seconds(
    *,
    rng: random.Random,
    min_s: float,
    max_s: float,
) -> float:
    """Sample the next mid-run sleep duration in ``[min_s, max_s]``.

    Pure given ``rng`` — guest clock is never used for security decisions;
    delays only pace challenge attempts.
    """
    if min_s < 0 or max_s < min_s:
        msg = f"invalid interval bounds: min_s={min_s} max_s={max_s}"
        raise ValueError(msg)
    if min_s == max_s:
        return float(min_s)
    return float(rng.uniform(min_s, max_s))


def phases_requiring_pre_delay(phases: Sequence[ChallengePhase]) -> frozenset[int]:
    """Indices that sleep before fetch (INTERVAL only; start/end are immediate)."""
    return frozenset(
        i for i, phase in enumerate(phases) if phase is ChallengePhase.INTERVAL
    )
