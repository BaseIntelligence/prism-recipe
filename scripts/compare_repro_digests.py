#!/usr/bin/env python3
"""Compare dual rebuild digests for the CI reproducibility gate.

Exit 0 only when rootfs content digests match (image Id match preferred).
On divergence, print RESULT=DIFF and name both digests so CI logs are actionable.

Used by scripts/repro-build-cpu.sh and .github/workflows/repro-image.yml.
"""

from __future__ import annotations

import argparse
import sys


def compare_digests(
    image_id_a: str,
    image_id_b: str,
    rootfs_a: str,
    rootfs_b: str,
) -> tuple[int, str]:
    """Return (exit_code, message). 0 = MATCH, 1 = DIFF."""
    lines: list[str] = [
        f"IMAGE_ID_A={image_id_a}",
        f"IMAGE_ID_B={image_id_b}",
        f"ROOTFS_CONTENT_DIGEST_A={rootfs_a}",
        f"ROOTFS_CONTENT_DIGEST_B={rootfs_b}",
    ]

    if not rootfs_a or not rootfs_b:
        lines.append("RESULT=DIFF empty content digest (export/hash failed)")
        return 1, "\n".join(lines) + "\n"

    if rootfs_a == rootfs_b and image_id_a == image_id_b:
        lines.append("RESULT=MATCH image Id and rootfs content digests identical")
        return 0, "\n".join(lines) + "\n"

    if rootfs_a == rootfs_b:
        lines.append(
            "RESULT=MATCH rootfs content digests identical "
            "(image Id may differ on config metadata)"
        )
        return 0, "\n".join(lines) + "\n"

    lines.append("RESULT=DIFF digests differ")
    lines.append(
        "CI reproducibility gate FAILED: rootfs content digests diverge "
        f"(A={rootfs_a} B={rootfs_b}). Non-deterministic build blocked."
    )
    return 1, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert two rebuild digests match (CI reproducibility gate)."
    )
    parser.add_argument("--image-id-a", required=True)
    parser.add_argument("--image-id-b", required=True)
    parser.add_argument("--rootfs-a", required=True)
    parser.add_argument("--rootfs-b", required=True)
    args = parser.parse_args(argv)
    code, msg = compare_digests(
        args.image_id_a,
        args.image_id_b,
        args.rootfs_a,
        args.rootfs_b,
    )
    sys.stdout.write(msg)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
