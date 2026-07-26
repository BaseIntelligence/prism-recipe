"""In-image attestation sidecar (mech 1+5+6 surface).

Runs inside the scored container. Answers BASE challenges at run start, run end,
and randomized intervals. Self-measures the sealed manifest at answer time,
signs with ``prism_recipe.attestation.payload``, and reads the per-build secret
from ``attestation_secret.ATTESTATION_HMAC_KEY_PATH``.

Honesty (binding):
* A valid signed answer proves only that an entity holding the in-image secret
  responded. It is **not** a hardware root of trust and is never sufficient
  alone for tier elevation.
* Do not claim independent verification or TEE.
"""

from __future__ import annotations

from prism_recipe.sidecar.config import SidecarConfig
from prism_recipe.sidecar.errors import SidecarError, SidecarReachabilityError
from prism_recipe.sidecar.schedule import next_interval_delay_seconds, plan_challenge_phases
from prism_recipe.sidecar.service import AttestationSidecar, ChallengeAnswer
from prism_recipe.sidecar.transport import FakeChallengeTransport
from prism_recipe.sidecar.types import Challenge, ChallengePhase
from prism_recipe.sidecar.wire import signed_attestation_to_wire

__all__ = [
    "AttestationSidecar",
    "Challenge",
    "ChallengeAnswer",
    "ChallengePhase",
    "FakeChallengeTransport",
    "SidecarConfig",
    "SidecarError",
    "SidecarReachabilityError",
    "next_interval_delay_seconds",
    "plan_challenge_phases",
    "signed_attestation_to_wire",
]
