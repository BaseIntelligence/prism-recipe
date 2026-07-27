"""CLI: ``python -m prism_recipe.sidecar``.

Exits non-zero when BASE is unreachable within the retry budget.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from prism_recipe.sidecar.config import SidecarConfig
from prism_recipe.sidecar.errors import SidecarError, SidecarReachabilityError
from prism_recipe.sidecar.listen import serve_forever
from prism_recipe.sidecar.service import AttestationSidecar
from prism_recipe.sidecar.transport import FakeChallengeTransport, HttpxChallengeTransport
from prism_recipe.sidecar.types import Challenge, ChallengePhase
from prism_recipe.sidecar.wire import signed_attestation_to_wire


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prism_recipe.sidecar",
        description=(
            "In-image attestation sidecar: answer BASE challenges with "
            "self-measured sealed-surface hashes (tamper-evidence, not TEE)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run start/interval/end challenge loop")
    p_run.add_argument(
        "--fake",
        action="store_true",
        help="Use in-process fake BASE (for local QA; not production)",
    )
    p_run.add_argument("--pod-id", default=None)
    p_run.add_argument("--digest", default=None)
    p_run.add_argument("--variant", choices=("cpu", "cuda"), default=None)
    p_run.add_argument("--secret-path", type=Path, default=None)
    p_run.add_argument("--root", type=Path, default=None)
    p_run.add_argument("--base-url", default=None)
    p_run.add_argument("--retry-budget", type=int, default=None)
    p_run.add_argument("--interval-count", type=int, default=None)
    p_run.add_argument("--interval-min-s", type=float, default=None)
    p_run.add_argument("--interval-max-s", type=float, default=None)
    p_run.add_argument("--seed", type=int, default=0)

    p_once = sub.add_parser("answer-once", help="Answer a single challenge JSON")
    p_once.add_argument("--nonce", required=True)
    p_once.add_argument("--phase", choices=("start", "interval", "end"), default="start")
    p_once.add_argument("--pod-id", required=True)
    p_once.add_argument("--digest", required=True)
    p_once.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    p_once.add_argument("--secret-path", type=Path, required=True)
    p_once.add_argument("--root", type=Path, default=None)

    p_serve = sub.add_parser(
        "serve",
        help="Listen for BASE attest requests (stdlib HTTP; no pull loop)",
    )
    p_serve.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (explicit; default 0.0.0.0 for container publish)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Bind port (default 8787; publish via Lium internal_ports)",
    )
    p_serve.add_argument("--pod-id", default=None)
    p_serve.add_argument("--digest", default=None)
    p_serve.add_argument("--variant", choices=("cpu", "cuda"), default=None)
    p_serve.add_argument("--secret-path", type=Path, default=None)
    p_serve.add_argument("--root", type=Path, default=None)

    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.command == "answer-once":
        return _cmd_answer_once(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "serve":
        return _cmd_serve(args)
    return 2


def _cmd_answer_once(args: argparse.Namespace) -> int:
    try:
        cfg = SidecarConfig(
            pod_id=args.pod_id,
            digest=args.digest,
            variant=args.variant,
            secret_path=args.secret_path,
            root=args.root,
            retry_budget=1,
            interval_count=0,
        )
        sidecar = AttestationSidecar(cfg)
        challenge = Challenge(
            nonce=args.nonce,
            phase=ChallengePhase(args.phase),
        )
        answer = sidecar.answer_challenge(challenge)
        wire = signed_attestation_to_wire(answer.signed)
        wire["phase"] = answer.phase.value
        wire["baked_manifest_match"] = answer.baked_manifest_match
        wire["mismatched_paths"] = list(answer.mismatched_paths)
        sys.stdout.write(json.dumps(wire, indent=2, sort_keys=True) + "\n")
        return 0
    except (SidecarError, SidecarReachabilityError, ValueError) as exc:
        sys.stderr.write(f"sidecar error: {exc}\n")
        return 1


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        if args.fake:
            cfg = _config_from_args_or_env(args, require_identity=True)
            transport: FakeChallengeTransport | HttpxChallengeTransport = FakeChallengeTransport(
                challenges_by_phase={
                    ChallengePhase.START: Challenge(
                        nonce="fake-nonce-start", phase=ChallengePhase.START
                    ),
                    ChallengePhase.INTERVAL: Challenge(
                        nonce="fake-nonce-interval", phase=ChallengePhase.INTERVAL
                    ),
                    ChallengePhase.END: Challenge(nonce="fake-nonce-end", phase=ChallengePhase.END),
                }
            )
        else:
            cfg = _config_from_args_or_env(args, require_identity=True)
            base_url = args.base_url or cfg.base_url
            if not base_url:
                raise SidecarError("base_url required (env PRISM_SIDECAR_BASE_URL)")
            transport = HttpxChallengeTransport(base_url=base_url)
        code = AttestationSidecar(cfg).run(transport=transport, rng=random.Random(args.seed))
        return code
    except SidecarError as exc:
        sys.stderr.write(f"sidecar error: {exc}\n")
        return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        cfg = _config_from_args_or_env(args, require_identity=True)
        # Attach optional CLI-only knobs that _config_from_args_or_env may omit
        # when falling back to env (base_url unused in listen mode).
        serve_forever(cfg, host=str(args.host), port=int(args.port))
        return 0
    except (SidecarError, SidecarReachabilityError, ValueError, OSError) as exc:
        sys.stderr.write(f"sidecar error: {exc}\n")
        return 1


def _config_from_args_or_env(args: argparse.Namespace, *, require_identity: bool) -> SidecarConfig:
    retry_budget = getattr(args, "retry_budget", None)
    interval_count = getattr(args, "interval_count", None)
    interval_min_s = getattr(args, "interval_min_s", None)
    interval_max_s = getattr(args, "interval_max_s", None)
    base_url = getattr(args, "base_url", None)
    if args.pod_id and args.digest:
        variant = args.variant or "cpu"
        secret = args.secret_path or SidecarConfig.default_secret_path()
        return SidecarConfig(
            pod_id=args.pod_id,
            digest=args.digest,
            variant=variant,
            secret_path=secret,
            root=args.root,
            retry_budget=retry_budget or 3,
            interval_count=(interval_count if interval_count is not None else 0),
            interval_min_s=interval_min_s if interval_min_s is not None else 0.0,
            interval_max_s=interval_max_s if interval_max_s is not None else 0.0,
            base_url=base_url,
        )
    if require_identity and not (args.pod_id and args.digest):
        # Fall back to env for container runs.
        env_cfg = SidecarConfig.from_env()
        return SidecarConfig(
            pod_id=args.pod_id or env_cfg.pod_id,
            digest=args.digest or env_cfg.digest,
            variant=args.variant or env_cfg.variant,
            secret_path=args.secret_path or env_cfg.secret_path,
            root=args.root,
            retry_budget=retry_budget or env_cfg.retry_budget,
            interval_count=(
                interval_count if interval_count is not None else env_cfg.interval_count
            ),
            interval_min_s=(
                interval_min_s if interval_min_s is not None else env_cfg.interval_min_s
            ),
            interval_max_s=(
                interval_max_s if interval_max_s is not None else env_cfg.interval_max_s
            ),
            base_url=base_url or env_cfg.base_url,
        )
    raise SidecarError("pod_id and digest required")


if __name__ == "__main__":
    raise SystemExit(main())
