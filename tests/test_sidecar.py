"""TDD: in-image attestation sidecar (checkbox 11).

Answers BASE challenges at start / random intervals / end. Self-measures the
sealed surface at answer time, signs with attestation.payload, reads the
build secret from attestation_secret path. Fake transport only — no live BASE.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from prism_recipe.attestation.payload import (
    derive_attestation_key,
    verify_attestation_payload,
)
from prism_recipe.attestation_secret import ATTESTATION_HMAC_KEY_PATH
from prism_recipe.sealed_surface import (
    bake_manifest,
    build_manifest,
    sha256_file,
)

BUILD_SECRET = b"sidecar-unit-test-build-secret-v1"
DIGEST = "sha256:" + ("cd" * 32)
POD_ID = "pod_sidecar_001"
NONCE_START = "11111111-1111-4111-8111-111111111111"
NONCE_INTERVAL = "22222222-2222-4222-8222-222222222222"
NONCE_END = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def recipe_root(tmp_path: Path) -> Path:
    """Minimal sealed-surface tree under a temp root (subset of real paths)."""
    # Use the real checkout as root so sealed files exist; bake manifest there
    # would pollute — instead point PRISM_RECIPE_HOME at real repo and bake to tmp.
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "prism_recipe").is_dir()
    return root


@pytest.fixture
def baked_manifest_path(recipe_root: Path, tmp_path: Path) -> Path:
    out = tmp_path / "sealed_surface_manifest.json"
    bake_manifest(recipe_root, out)
    return out


@pytest.fixture
def secret_path(tmp_path: Path) -> Path:
    p = tmp_path / "attestation_hmac_key"
    p.write_bytes(BUILD_SECRET)
    p.chmod(0o400)
    return p


def _import_sidecar():
    from prism_recipe.sidecar import (
        AttestationSidecar,
        Challenge,
        ChallengePhase,
        FakeChallengeTransport,
        SidecarConfig,
        SidecarError,
        SidecarReachabilityError,
        next_interval_delay_seconds,
        plan_challenge_phases,
        signed_attestation_to_wire,
    )

    return {
        "Challenge": Challenge,
        "ChallengePhase": ChallengePhase,
        "FakeChallengeTransport": FakeChallengeTransport,
        "SidecarConfig": SidecarConfig,
        "SidecarError": SidecarError,
        "SidecarReachabilityError": SidecarReachabilityError,
        "AttestationSidecar": AttestationSidecar,
        "next_interval_delay_seconds": next_interval_delay_seconds,
        "plan_challenge_phases": plan_challenge_phases,
        "signed_attestation_to_wire": signed_attestation_to_wire,
    }


def test_plan_phases_start_intervals_end() -> None:
    """S3: schedule is start, then N intervals, then end — pure + deterministic."""
    sc = _import_sidecar()
    phases = sc["plan_challenge_phases"](interval_count=3)
    assert [p.value for p in phases] == [
        "start",
        "interval",
        "interval",
        "interval",
        "end",
    ]


def test_next_interval_delay_is_within_bounds_and_deterministic() -> None:
    """S3b: random interval delay stays in [min, max] for a fixed RNG seed."""
    sc = _import_sidecar()
    rng = random.Random(42)
    delays = [
        sc["next_interval_delay_seconds"](rng=rng, min_s=10.0, max_s=30.0)
        for _ in range(20)
    ]
    assert all(10.0 <= d <= 30.0 for d in delays)
    rng2 = random.Random(42)
    delays2 = [
        sc["next_interval_delay_seconds"](rng=rng2, min_s=10.0, max_s=30.0)
        for _ in range(20)
    ]
    assert delays == delays2


def test_answer_start_challenge_verifies_and_matches_baked(
    recipe_root: Path,
    baked_manifest_path: Path,
    secret_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1 happy: challenge → signed answer verifies; live hashes == baked."""
    sc = _import_sidecar()
    monkeypatch.setenv("PRISM_RECIPE_SEALED_MANIFEST", str(baked_manifest_path))
    monkeypatch.setenv("PRISM_RECIPE_HOME", str(recipe_root))

    cfg = sc["SidecarConfig"](
        pod_id=POD_ID,
        digest=DIGEST,
        variant="cpu",
        secret_path=secret_path,
        root=recipe_root,
        retry_budget=3,
        interval_count=0,
        interval_min_s=1.0,
        interval_max_s=2.0,
    )
    sidecar = sc["AttestationSidecar"](cfg)
    challenge = sc["Challenge"](
        nonce=NONCE_START,
        phase=sc["ChallengePhase"].START,
    )

    answer = sidecar.answer_challenge(challenge)

    verify_key = derive_attestation_key(BUILD_SECRET)
    result = verify_attestation_payload(answer.signed, verify_key=verify_key)
    assert result.ok is True
    assert answer.signed.payload.nonce == NONCE_START
    assert answer.signed.payload.pod_id == POD_ID
    assert answer.signed.payload.digest == DIGEST
    assert answer.signed.payload.variant == "cpu"
    assert answer.baked_manifest_match is True
    assert answer.mismatched_paths == ()

    baked = json.loads(baked_manifest_path.read_text(encoding="utf-8"))
    assert dict(answer.signed.payload.sealed_manifest_hashes) == baked["files"]


def test_run_start_end_posts_verifiable_answers(
    recipe_root: Path,
    baked_manifest_path: Path,
    secret_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1b: full run with fake transport posts start+end answers BASE can verify."""
    sc = _import_sidecar()
    monkeypatch.setenv("PRISM_RECIPE_SEALED_MANIFEST", str(baked_manifest_path))
    monkeypatch.setenv("PRISM_RECIPE_HOME", str(recipe_root))

    transport = sc["FakeChallengeTransport"](
        challenges_by_phase={
            sc["ChallengePhase"].START: sc["Challenge"](
                nonce=NONCE_START, phase=sc["ChallengePhase"].START
            ),
            sc["ChallengePhase"].END: sc["Challenge"](
                nonce=NONCE_END, phase=sc["ChallengePhase"].END
            ),
        }
    )
    cfg = sc["SidecarConfig"](
        pod_id=POD_ID,
        digest=DIGEST,
        variant="cpu",
        secret_path=secret_path,
        root=recipe_root,
        retry_budget=2,
        interval_count=0,
        interval_min_s=0.0,
        interval_max_s=0.0,
    )
    code = sc["AttestationSidecar"](cfg).run(
        transport=transport, rng=random.Random(0), sleep_fn=lambda _s: None
    )
    assert code == 0
    assert len(transport.submitted) == 2
    verify_key = derive_attestation_key(BUILD_SECRET)
    for wire in transport.submitted:
        signed = sc["AttestationSidecar"].signed_from_wire(wire)
        assert verify_attestation_payload(signed, verify_key=verify_key).ok is True


def test_tamper_sealed_file_reports_hash_mismatch(
    recipe_root: Path,
    baked_manifest_path: Path,
    secret_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: mutate a sealed file → answer hashes diverge from baked (tamper evidence)."""
    sc = _import_sidecar()
    # Work on a copy of one sealed file under a shadow root is hard; instead
    # monkeypatch sha256_file for one path after baking, or copy tree.
    # Practical approach: bake from real root, then answer with a root that has
    # one file overwritten via a temp overlay by patching measure to use a
    # mutated file map — better: copy one sealed file into tmp and point
    # measure via monkeypatch of sealed_surface.sha256_file for that path.

    baked = json.loads(baked_manifest_path.read_text(encoding="utf-8"))
    target_rel = "src/prism_recipe/harness.py"
    original_hash = baked["files"][target_rel]
    target_path = recipe_root / target_rel
    original_bytes = target_path.read_bytes()
    mutated = original_bytes + b"\n# sidecar-tamper-probe\n"
    assert sha256_file.__module__  # imported

    def _sha256_maybe_tampered(path: Path) -> str:
        if path.resolve() == target_path.resolve():
            import hashlib

            return hashlib.sha256(mutated).hexdigest()
        return sha256_file(path)

    monkeypatch.setenv("PRISM_RECIPE_SEALED_MANIFEST", str(baked_manifest_path))
    monkeypatch.setenv("PRISM_RECIPE_HOME", str(recipe_root))
    monkeypatch.setattr(
        "prism_recipe.sidecar.measure.sha256_file", _sha256_maybe_tampered
    )
    # Also patch sealed_surface used if measure re-exports differently
    monkeypatch.setattr(
        "prism_recipe.sealed_surface.sha256_file", _sha256_maybe_tampered
    )

    cfg = sc["SidecarConfig"](
        pod_id=POD_ID,
        digest=DIGEST,
        variant="cpu",
        secret_path=secret_path,
        root=recipe_root,
        retry_budget=1,
        interval_count=0,
        interval_min_s=1.0,
        interval_max_s=1.0,
    )
    answer = sc["AttestationSidecar"](cfg).answer_challenge(
        sc["Challenge"](nonce=NONCE_INTERVAL, phase=sc["ChallengePhase"].INTERVAL)
    )

    live = answer.signed.payload.sealed_manifest_hashes[target_rel]
    assert live != original_hash
    assert answer.baked_manifest_match is False
    assert target_rel in answer.mismatched_paths
    # Signature still valid over the *live* (tampered) measurement
    verify_key = derive_attestation_key(BUILD_SECRET)
    assert verify_attestation_payload(answer.signed, verify_key=verify_key).ok is True


def test_retry_budget_exhausted_exits_nonzero(
    recipe_root: Path,
    baked_manifest_path: Path,
    secret_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4: cannot reach BASE within retry budget → non-zero exit / typed error."""
    sc = _import_sidecar()
    monkeypatch.setenv("PRISM_RECIPE_SEALED_MANIFEST", str(baked_manifest_path))
    monkeypatch.setenv("PRISM_RECIPE_HOME", str(recipe_root))

    class DeadTransport:
        def fetch_challenge(self, *, phase: Any) -> Any:
            raise sc["SidecarReachabilityError"]("base unreachable")

        def submit_answer(self, answer: dict[str, Any]) -> None:
            raise sc["SidecarReachabilityError"]("base unreachable")

    cfg = sc["SidecarConfig"](
        pod_id=POD_ID,
        digest=DIGEST,
        variant="cpu",
        secret_path=secret_path,
        root=recipe_root,
        retry_budget=2,
        interval_count=0,
        interval_min_s=0.0,
        interval_max_s=0.0,
    )
    code = sc["AttestationSidecar"](cfg).run(
        transport=DeadTransport(),
        rng=random.Random(0),
        sleep_fn=lambda _s: None,
    )
    assert code != 0


def test_missing_secret_fails_closed(
    recipe_root: Path,
    baked_manifest_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge: missing hmac key path → typed SidecarError, no invented secret."""
    sc = _import_sidecar()
    monkeypatch.setenv("PRISM_RECIPE_SEALED_MANIFEST", str(baked_manifest_path))
    missing = tmp_path / "no-key"
    cfg = sc["SidecarConfig"](
        pod_id=POD_ID,
        digest=DIGEST,
        variant="cpu",
        secret_path=missing,
        root=recipe_root,
        retry_budget=1,
        interval_count=0,
        interval_min_s=1.0,
        interval_max_s=1.0,
    )
    with pytest.raises(sc["SidecarError"]):
        sc["AttestationSidecar"](cfg).answer_challenge(
            sc["Challenge"](nonce=NONCE_START, phase=sc["ChallengePhase"].START)
        )


def test_default_secret_path_constant_matches_mech6() -> None:
    """Adjacent: sidecar default key path is the mech-6 baked path."""
    sc = _import_sidecar()
    assert sc["SidecarConfig"].default_secret_path() == Path(ATTESTATION_HMAC_KEY_PATH)


def test_wire_format_roundtrip_fields(
    recipe_root: Path,
    baked_manifest_path: Path,
    secret_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire JSON carries schema/algorithm/signature/payload for BASE consume."""
    sc = _import_sidecar()
    monkeypatch.setenv("PRISM_RECIPE_SEALED_MANIFEST", str(baked_manifest_path))
    monkeypatch.setenv("PRISM_RECIPE_HOME", str(recipe_root))
    cfg = sc["SidecarConfig"](
        pod_id=POD_ID,
        digest=DIGEST,
        variant="cuda",
        secret_path=secret_path,
        root=recipe_root,
        retry_budget=1,
        interval_count=0,
        interval_min_s=1.0,
        interval_max_s=1.0,
    )
    answer = sc["AttestationSidecar"](cfg).answer_challenge(
        sc["Challenge"](nonce=NONCE_START, phase=sc["ChallengePhase"].START)
    )
    wire = sc["signed_attestation_to_wire"](answer.signed)
    assert wire["schema_version"] == "prism_attestation_payload.v1"
    assert wire["algorithm"] == "hmac-sha256"
    assert isinstance(wire["signature"], str) and len(wire["signature"]) == 64
    assert wire["payload"]["nonce"] == NONCE_START
    assert wire["payload"]["variant"] == "cuda"
    assert "sealed_manifest_hashes" in wire["payload"]
    # Honesty markers for evidence dumps (not part of signed body)
    assert wire.get("hardware_root_of_trust") is False
    assert wire.get("sufficient_alone_for_tier_elevation") is False


def test_build_manifest_fixture_nonempty(recipe_root: Path) -> None:
    """Sanity: sealed surface under recipe root is measurable."""
    m = build_manifest(recipe_root)
    assert len(m["files"]) >= 5
