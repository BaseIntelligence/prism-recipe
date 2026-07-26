"""TDD: pure comparison gate for dual CPU image rebuild digests.

The CI reproducibility job builds twice and must fail loudly when rootfs
content digests diverge (e.g. a timestamp baked into the image).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "compare_repro_digests.py"


def _load_helper():
    assert HELPER.is_file(), f"missing helper {HELPER}"
    spec = importlib.util.spec_from_file_location("compare_repro_digests", HELPER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_match_when_image_id_and_rootfs_identical() -> None:
    """Happy: same image Id + same rootfs content digest → MATCH exit 0."""
    mod = _load_helper()
    img = "sha256:d4b9d860ee56393f63848513d82d871a2b7dcc083fa5a8179446f520df7ba5ea"
    rootfs = "44e3ac3ea276ba39edcd5a4b187b0e508ea4153ec528666a0051cd7d754ed991"
    code, msg = mod.compare_digests(img, img, rootfs, rootfs)
    assert code == 0
    assert "RESULT=MATCH" in msg
    assert rootfs in msg


def test_match_when_rootfs_identical_even_if_image_id_differs() -> None:
    """Edge: rootfs content is the security-relevant digest; Id may differ on config metadata."""
    mod = _load_helper()
    rootfs = "44e3ac3ea276ba39edcd5a4b187b0e508ea4153ec528666a0051cd7d754ed991"
    code, msg = mod.compare_digests("sha256:aaa", "sha256:bbb", rootfs, rootfs)
    assert code == 0
    assert "RESULT=MATCH" in msg
    assert "rootfs" in msg.lower()


def test_diff_names_both_rootfs_digests_loudly() -> None:
    """Failure: timestamp-injected rebuild diverges — message must name BOTH digests."""
    mod = _load_helper()
    a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    code, msg = mod.compare_digests("sha256:id-a", "sha256:id-b", a, b)
    assert code == 1
    assert "RESULT=DIFF" in msg
    assert a in msg
    assert b in msg
    # Loud naming for operators / CI logs
    assert "ROOTFS_CONTENT_DIGEST_A=" in msg
    assert "ROOTFS_CONTENT_DIGEST_B=" in msg


def test_empty_digest_is_diff() -> None:
    """Edge: empty export/hash must not silently MATCH."""
    mod = _load_helper()
    code, msg = mod.compare_digests("sha256:x", "sha256:x", "", "deadbeef")
    assert code == 1
    assert "RESULT=DIFF" in msg
    assert "empty" in msg.lower()


def test_cli_exit_codes_match_and_diff() -> None:
    """Surface: CLI used by repro-build-cpu.sh / CI.

    Exit 0 MATCH, exit 1 DIFF with both digests named.
    """
    img = "sha256:d4b9d860ee56393f63848513d82d871a2b7dcc083fa5a8179446f520df7ba5ea"
    rootfs = "44e3ac3ea276ba39edcd5a4b187b0e508ea4153ec528666a0051cd7d754ed991"
    ok = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--image-id-a",
            img,
            "--image-id-b",
            img,
            "--rootfs-a",
            rootfs,
            "--rootfs-b",
            rootfs,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr + ok.stdout
    assert "RESULT=MATCH" in ok.stdout

    bad_a = "1111111111111111111111111111111111111111111111111111111111111111"
    bad_b = "2222222222222222222222222222222222222222222222222222222222222222"
    bad = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--image-id-a",
            "sha256:a",
            "--image-id-b",
            "sha256:b",
            "--rootfs-a",
            bad_a,
            "--rootfs-b",
            bad_b,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "RESULT=DIFF" in bad.stdout
    assert bad_a in bad.stdout
    assert bad_b in bad.stdout


def test_workflow_invokes_repro_build_cpu_script() -> None:
    """Adjacent: CI gate must call the proven hermetic dual-build script, not a weaker path."""
    wf = ROOT / ".github" / "workflows" / "repro-image.yml"
    assert wf.is_file(), f"missing workflow {wf}"
    text = wf.read_text(encoding="utf-8")
    assert "scripts/repro-build-cpu.sh" in text
    assert "SOURCE_DATE_EPOCH" in text or "1704067200" in text
    assert "provenance" in text.lower() or "repro-build-cpu" in text
    # Must not invent a weaker single-build path as the only gate
    assert "repro-build-cpu.sh" in text


@pytest.mark.parametrize(
    "fragment",
    [
        "rewrite-timestamp=true",
        "--provenance=false",
        "--sbom=false",
        "SOURCE_DATE_EPOCH",
    ],
)
def test_repro_build_script_keeps_hermetic_flags(fragment: str) -> None:
    """Regression: dual-build script must retain proven hermetic buildx flags."""
    script = (ROOT / "scripts" / "repro-build-cpu.sh").read_text(encoding="utf-8")
    assert fragment in script
