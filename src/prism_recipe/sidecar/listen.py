"""Stdlib HTTP listen mode for the in-image attestation sidecar.

BASE dials the running instance and POSTs a fresh nonce. No third-party
server stack — ``http.server.ThreadingHTTPServer`` only (hermetic image).
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import urlparse

from prism_recipe.sidecar.config import SidecarConfig
from prism_recipe.sidecar.errors import SidecarError
from prism_recipe.sidecar.service import AttestationSidecar, ChallengeAnswer
from prism_recipe.sidecar.types import Challenge, ChallengePhase
from prism_recipe.sidecar.wire import signed_attestation_to_wire

logger = logging.getLogger(__name__)

ATTEST_PATH: Final[str] = "/v1/sidecar/attest"
HEALTHZ_PATH: Final[str] = "/healthz"
MAX_BODY_BYTES: Final[int] = 64 * 1024
_VALID_PHASES: Final[frozenset[str]] = frozenset({"start", "interval", "end"})


def build_server(
    config: SidecarConfig,
    *,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    """Bind an explicit host/port and return a ready ``ThreadingHTTPServer``.

    ``host`` must be caller-provided (no implicit ``0.0.0.0`` default here).
    Pass ``port=0`` in tests for an ephemeral port.
    """
    if not host.strip():
        msg = "host must be non-empty"
        raise ValueError(msg)
    sidecar = AttestationSidecar(config)
    handler = _make_handler(sidecar)
    return ThreadingHTTPServer((host, port), handler)


def serve_forever(config: SidecarConfig, *, host: str, port: int) -> None:
    """Build the server and block serving until interrupted."""
    server = build_server(config, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    logger.info("sidecar listen mode on http://%s:%s", bound_host, bound_port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def answer_to_wire(answer: ChallengeAnswer) -> dict[str, Any]:
    """Serialize a ``ChallengeAnswer`` to the answer-once wire shape."""
    wire = signed_attestation_to_wire(answer.signed)
    wire["phase"] = answer.phase.value
    wire["baked_manifest_match"] = answer.baked_manifest_match
    wire["mismatched_paths"] = list(answer.mismatched_paths)
    return wire


def _make_handler(sidecar: AttestationSidecar) -> type[BaseHTTPRequestHandler]:
    class SidecarHTTPRequestHandler(BaseHTTPRequestHandler):
        server_version = "PrismSidecar/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            # Access log only — never include body/secret material.
            logger.info("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802 — stdlib handler API
            path = urlparse(self.path).path
            if path == HEALTHZ_PATH:
                self._json_response(200, {"status": "ok"})
                return
            if path == ATTEST_PATH:
                self._json_response(405, {"error": "method not allowed"})
                return
            self._json_response(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 — stdlib handler API
            path = urlparse(self.path).path
            if path != ATTEST_PATH:
                if path == HEALTHZ_PATH:
                    self._json_response(405, {"error": "method not allowed"})
                    return
                self._json_response(404, {"error": "not found"})
                return
            self._handle_attest()

        def do_PUT(self) -> None:  # noqa: N802
            self._method_not_allowed_or_404()

        def do_DELETE(self) -> None:  # noqa: N802
            self._method_not_allowed_or_404()

        def do_PATCH(self) -> None:  # noqa: N802
            self._method_not_allowed_or_404()

        def _method_not_allowed_or_404(self) -> None:
            path = urlparse(self.path).path
            if path in {ATTEST_PATH, HEALTHZ_PATH}:
                self._json_response(405, {"error": "method not allowed"})
                return
            self._json_response(404, {"error": "not found"})

        def _handle_attest(self) -> None:
            try:
                raw = self._read_body()
            except _ClientBodyError as exc:
                self._json_response(exc.status, {"error": exc.message})
                return
            try:
                challenge = _parse_attest_body(raw)
            except ValueError as exc:
                self._json_response(400, {"error": str(exc)})
                return
            try:
                answer = sidecar.answer_challenge(challenge)
                wire = answer_to_wire(answer)
            except SidecarError:
                logger.exception("sidecar attest failed")
                self._json_response(500, {"error": "internal error"})
                return
            except Exception:
                logger.exception("unexpected attest failure")
                self._json_response(500, {"error": "internal error"})
                return
            self._json_response(200, wire)

        def _read_body(self) -> bytes:
            length_hdr = self.headers.get("Content-Length")
            if length_hdr is None:
                # No body / chunked not supported — treat as empty.
                return b""
            try:
                length = int(length_hdr)
            except ValueError as exc:
                raise _ClientBodyError(400, "invalid Content-Length") from exc
            if length < 0:
                raise _ClientBodyError(400, "invalid Content-Length")
            if length > MAX_BODY_BYTES:
                raise _ClientBodyError(413, "request body too large")
            return self.rfile.read(length)

        def _json_response(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

    return SidecarHTTPRequestHandler


def _parse_attest_body(raw: bytes) -> Challenge:
    """Parse POST /v1/sidecar/attest JSON into a ``Challenge`` (boundary)."""
    if not raw.strip():
        msg = "request body required"
        raise ValueError(msg)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = "malformed JSON"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = "JSON body must be an object"
        raise ValueError(msg)
    nonce = data.get("nonce")
    if not isinstance(nonce, str) or not nonce.strip():
        msg = "nonce must be a non-empty string"
        raise ValueError(msg)
    phase_raw = data.get("phase", "start")
    if not isinstance(phase_raw, str):
        msg = "phase must be a string"
        raise ValueError(msg)
    phase_key = phase_raw.strip().lower()
    if phase_key not in _VALID_PHASES:
        msg = f"invalid phase: {phase_raw!r}"
        raise ValueError(msg)
    match phase_key:
        case "start":
            phase = ChallengePhase.START
        case "interval":
            phase = ChallengePhase.INTERVAL
        case "end":
            phase = ChallengePhase.END
        case unreachable:
            msg = f"invalid phase: {unreachable!r}"
            raise ValueError(msg)
    return Challenge(nonce=nonce.strip(), phase=phase)


class _ClientBodyError(Exception):
    """Client-caused body read failure (maps to 4xx)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
