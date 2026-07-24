"""CLI entry for recipe image workers."""

from __future__ import annotations

import argparse
import json
import sys

from prism_recipe import __version__
from prism_recipe.harness import preflight, run_train
from prism_recipe.llm_gate import rules_digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prism-recipe",
        description="PRISM recipe harness (egalitarian data + LLM gate + train)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="Run env + LLM gate without training")
    p_train = sub.add_parser(
        "train",
        help="Run recipe train path (use --smoke for offline tiny-1m)",
    )
    p_train.add_argument(
        "--smoke",
        action="store_true",
        help="Offline tiny-1m smoke train (fixture data, small budget)",
    )
    p_train.add_argument(
        "--gpu-short",
        action="store_true",
        help="Real CUDA short train (~1M params, 50k-200k tokens) for score path",
    )
    p_gpu = sub.add_parser(
        "gpu-short",
        help="Alias for train --gpu-short (real CUDA ~1M short train)",
    )
    p_gpu.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Token budget (default 65536; clamp 1024..200000)",
    )
    p_smoke = sub.add_parser(
        "smoke",
        help="Alias for train --smoke (offline end-to-end tiny-1m)",
    )
    p_smoke.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip live OpenRouter (local CI); still records gate attestation metadata",
    )
    p_digest = sub.add_parser("rules-digest", help="Print SHA-256 of bundled .rules/")
    p_digest.add_argument("--json", action="store_true", help="Emit JSON")

    args = parser.parse_args(argv)

    if args.command == "rules-digest":
        digest = rules_digest()
        if args.json:
            print(json.dumps({"rules_digest": digest}))
        else:
            print(digest)
        return 0

    if args.command == "preflight":
        outcome = preflight()
        print(json.dumps(outcome.as_dict(), indent=2))
        return 0 if outcome.ok else 2

    if args.command == "smoke":
        import os

        os.environ["PRISM_RECIPE_SMOKE"] = "1"
        if getattr(args, "skip_gate", False):
            os.environ["PRISM_RECIPE_SMOKE_SKIP_GATE"] = "1"
        if not os.environ.get("PRISM_RECIPE_TOKEN_BUDGET"):
            os.environ["PRISM_RECIPE_TOKEN_BUDGET"] = "256"
        outcome = run_train(smoke=True)
        print(json.dumps(outcome.as_dict(), indent=2))
        return 0 if outcome.ok else 2

    if args.command == "gpu-short":
        import os

        os.environ["PRISM_RECIPE_GPU_SHORT"] = "1"
        # Scoring path: never skip gate unless caller already set env (CI only).
        budget = getattr(args, "token_budget", None)
        if budget is not None:
            os.environ["PRISM_RECIPE_TOKEN_BUDGET"] = str(int(budget))
        elif not os.environ.get("PRISM_RECIPE_TOKEN_BUDGET"):
            os.environ["PRISM_RECIPE_TOKEN_BUDGET"] = "65536"
        outcome = run_train(smoke=False)
        print(json.dumps(outcome.as_dict(), indent=2))
        return 0 if outcome.ok else 2

    if args.command == "train":
        import os

        smoke = bool(getattr(args, "smoke", False))
        gpu_short = bool(getattr(args, "gpu_short", False))
        if gpu_short:
            os.environ["PRISM_RECIPE_GPU_SHORT"] = "1"
            if not os.environ.get("PRISM_RECIPE_TOKEN_BUDGET"):
                os.environ["PRISM_RECIPE_TOKEN_BUDGET"] = "65536"
            smoke = False
        if smoke:
            os.environ["PRISM_RECIPE_SMOKE"] = "1"
            if not os.environ.get("PRISM_RECIPE_TOKEN_BUDGET"):
                os.environ["PRISM_RECIPE_TOKEN_BUDGET"] = "256"
        outcome = run_train(smoke=smoke or None)
        print(json.dumps(outcome.as_dict(), indent=2))
        return 0 if outcome.ok else 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
