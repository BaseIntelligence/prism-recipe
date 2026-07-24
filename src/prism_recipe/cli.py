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
    sub.add_parser("train", help="Run full recipe train path (stub until later features)")
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

    if args.command == "train":
        outcome = run_train()
        print(json.dumps(outcome.as_dict(), indent=2))
        return 0 if outcome.ok else 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
