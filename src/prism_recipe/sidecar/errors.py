"""Typed errors for the in-image attestation sidecar."""

from __future__ import annotations


class SidecarError(RuntimeError):
    """Fail-closed sidecar failure (secret, measure, config, or protocol)."""


class SidecarReachabilityError(SidecarError):
    """BASE challenge endpoint unreachable or rejected the transport call."""
