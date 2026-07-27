"""TDD: sidecar listen/serve mode (stdlib HTTP server).

BASE dials the running instance and POSTs a fresh nonce; the sidecar answers
with the same wire shape as ``answer-once``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from prism_recipe.attestation.payload import (
    derive_attestation_key,
    verify_attestation_payload,
)
from prism_recipe.sealed_surface import bake_manifest
from prism_recipe.sidecar.config import SidecarConfig
from prism_recipe.sidecar.service import AttestationSidecar
from prism_recipe.sidecar.types import Challenge, ChallengePhase
from prism_recipe.sidecar.wire import signed_attestation_to_wire, signed_from_wire

BUILD_SECRET = b"sidecar-listen-unit-test-build-secret-v1"
DIGEST = "sha256:" + ("ab" * 32)
POD_ID = "pod_listen_001"
NONCE = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
MAX_BODY = 64 * 1024


@pytest.fixture
def recipe_root() -> Path:
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
    path = tmp_path / "attestation_hmac_key"
    path.write_bytes(BUILD_SECRET)
    path.chmod(0o400)
    return path


@pytest.fixture
def sidecar_config(
    recipe_root: Path,
    secret_path: Path,
    baked_manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SidecarConfig:
    monkeypatch.setenv("PRISM_RECIPE_SEALED_MANIFEST", str(baked_manifest_path))
    monkeypatch.setenv("PRISM_RECIPE_HOME", str(recipe_root))
    return SidecarConfig(
        pod_id=POD_ID,
        digest=DIGEST,
        variant="cpu",
        secret_path=secret_path,
        root=recipe_root,
        retry_budget=1,
        interval_count=0,
    )


@pytest.fixture
def listen_server(sidecar_config: SidecarConfig) -> Iterator[tuple[ThreadingHTTPServer, str]]:
    from prism_recipe.sidecar.listen import build_server

    server = build_server(sidecar_config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        yield server, base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_healthz_ok(listen_server: tuple[ThreadingHTTPServer, str]) -> None:
    """Given running server; When GET /healthz; Then 200 JSON liveness."""
    _server, base = listen_server
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{base}/healthz")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("application/json")
    body = resp.json()
    assert body.get("status") == "ok"


def test_listen_attest_returns_verifiable_signature(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given POST attest with nonce; When BASE verifies HMAC; Then ok + digest."""
    _server, base = listen_server
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            json={"nonce": NONCE, "phase": "start"},
        )
    assert resp.status_code == 200
    wire = resp.json()
    signed = signed_from_wire(wire)
    verify_key = derive_attestation_key(BUILD_SECRET)
    result = verify_attestation_payload(signed, verify_key=verify_key)
    assert result.ok is True
    assert signed.payload.digest == DIGEST
    assert signed.payload.pod_id == POD_ID


def test_listen_attest_reflects_requested_nonce(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given a specific nonce; When attest; Then signed payload binds that nonce."""
    _server, base = listen_server
    nonce = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            json={"nonce": nonce, "phase": "interval"},
        )
    assert resp.status_code == 200
    wire = resp.json()
    assert wire["payload"]["nonce"] == nonce
    assert wire["phase"] == "interval"


def test_unknown_path_404(listen_server: tuple[ThreadingHTTPServer, str]) -> None:
    """Given unknown path; When GET; Then 404 JSON."""
    _server, base = listen_server
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{base}/nope")
    assert resp.status_code == 404
    assert resp.headers.get("content-type", "").startswith("application/json")
    assert "error" in resp.json()


def test_wrong_method_405(listen_server: tuple[ThreadingHTTPServer, str]) -> None:
    """Given known path wrong method; When GET attest; Then 405 JSON."""
    _server, base = listen_server
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{base}/v1/sidecar/attest")
    assert resp.status_code == 405
    assert resp.headers.get("content-type", "").startswith("application/json")
    assert "error" in resp.json()


def test_bad_json_returns_400_not_500(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given malformed JSON body; When POST attest; Then 400 not 500."""
    _server, base = listen_server
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            content=b"not-json{",
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_missing_nonce_returns_400(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given body without nonce; When POST attest; Then 400."""
    _server, base = listen_server
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            json={"phase": "start"},
        )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_blank_nonce_returns_400(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given blank nonce; When POST attest; Then 400."""
    _server, base = listen_server
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            json={"nonce": "   ", "phase": "start"},
        )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_invalid_phase_returns_400(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given invalid phase; When POST attest; Then 400."""
    _server, base = listen_server
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            json={"nonce": NONCE, "phase": "bogus"},
        )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_error_response_does_not_leak_secret(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given any error path; When inspecting bodies; Then secret never appears."""
    _server, base = listen_server
    secret_hex = BUILD_SECRET.hex()
    probes: list[httpx.Response] = []
    with httpx.Client(timeout=5.0) as client:
        probes.append(client.post(f"{base}/v1/sidecar/attest", content=b"{"))
        probes.append(client.post(f"{base}/v1/sidecar/attest", json={"phase": "start"}))
        probes.append(
            client.post(
                f"{base}/v1/sidecar/attest",
                json={"nonce": NONCE, "phase": "nope"},
            )
        )
        probes.append(client.get(f"{base}/missing"))
        probes.append(client.get(f"{base}/v1/sidecar/attest"))
        # Happy path body must not echo the raw secret either.
        probes.append(
            client.post(
                f"{base}/v1/sidecar/attest",
                json={"nonce": NONCE, "phase": "start"},
            )
        )
    for resp in probes:
        text = resp.text
        assert BUILD_SECRET.decode("ascii") not in text
        assert secret_hex not in text
        assert BUILD_SECRET not in resp.content


def test_wire_shape_matches_answer_once(
    listen_server: tuple[ThreadingHTTPServer, str],
    sidecar_config: SidecarConfig,
) -> None:
    """Given same challenge; When listen vs answer-once; Then JSON keys match."""
    _server, base = listen_server
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            json={"nonce": NONCE, "phase": "end"},
        )
    assert resp.status_code == 200
    listen_wire: dict[str, Any] = resp.json()

    sidecar = AttestationSidecar(sidecar_config)
    answer = sidecar.answer_challenge(Challenge(nonce=NONCE, phase=ChallengePhase.END))
    once_wire = signed_attestation_to_wire(answer.signed)
    once_wire["phase"] = answer.phase.value
    once_wire["baked_manifest_match"] = answer.baked_manifest_match
    once_wire["mismatched_paths"] = list(answer.mismatched_paths)

    assert set(listen_wire.keys()) == set(once_wire.keys())
    assert set(listen_wire["payload"].keys()) == set(once_wire["payload"].keys())
    for key in (
        "schema_version",
        "algorithm",
        "signature",
        "phase",
        "baked_manifest_match",
        "mismatched_paths",
        "hardware_root_of_trust",
        "sufficient_alone_for_tier_elevation",
        "proves",
    ):
        assert key in listen_wire
    assert listen_wire["payload"]["nonce"] == once_wire["payload"]["nonce"]
    assert listen_wire["payload"]["digest"] == once_wire["payload"]["digest"]
    assert listen_wire["signature"] == once_wire["signature"]


def test_oversized_body_rejected(
    listen_server: tuple[ThreadingHTTPServer, str],
) -> None:
    """Given body larger than max; When POST attest; Then 400/413 not 500."""
    _server, base = listen_server
    huge = b'{"nonce":"' + (b"x" * (MAX_BODY + 8)) + b'","phase":"start"}'
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(
            f"{base}/v1/sidecar/attest",
            content=huge,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code in {400, 413}
    assert resp.status_code != 500
    assert "error" in resp.json()
