"""Egalitarian FineWeb loader unit tests (mocked/tiny stream, no network)."""

from __future__ import annotations

import pytest

from prism_recipe.config import (
    EQUAL_OFFSET,
    PROD_DATASET_ID,
    PROD_DATASET_REVISION,
    PROD_TOKEN_BUDGET,
    TOKEN_BUDGET_PROD,
    DataWindowPin,
    prod_data_window,
    resolve_token_budget,
)
from prism_recipe.loader import (
    EgalitarianFineWebLoader,
    LoaderError,
    apply_offset_and_budget,
    build_loader,
    count_tokens,
    default_encode,
    iter_budgeted_tokens,
)


def _docs(*texts: str) -> list[dict[str, str]]:
    return [{"id": f"d{i}", "text": t} for i, t in enumerate(texts)]


def _encode_identity_words(text: str) -> list[int]:
    """Deterministic encoder: token id = 1..n per whitespace word order in doc."""
    return list(range(1, len(text.split()) + 1)) if text.strip() else []


def test_prod_pins_constants() -> None:
    assert TOKEN_BUDGET_PROD == 2_500_000_000
    assert TOKEN_BUDGET_PROD == PROD_TOKEN_BUDGET
    assert EQUAL_OFFSET == 0
    pin = prod_data_window()
    assert pin.dataset_id == PROD_DATASET_ID == "HuggingFaceFW/fineweb-edu"
    assert pin.dataset_revision == PROD_DATASET_REVISION
    assert len(pin.dataset_revision) == 40  # immutable commit SHA, not "main"
    assert pin.token_start_offset == EQUAL_OFFSET
    assert pin.token_budget == TOKEN_BUDGET_PROD
    assert pin.epochs == 1
    assert pin.identity_tuple() == (
        PROD_DATASET_ID,
        PROD_DATASET_REVISION,
        EQUAL_OFFSET,
    )


def test_smoke_budget_env_does_not_change_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_RECIPE_TOKEN_BUDGET", "1000000")
    assert resolve_token_budget() == 1_000_000
    plan = build_loader().plan()
    assert plan.token_budget == 1_000_000
    assert plan.pin.token_start_offset == EQUAL_OFFSET
    assert plan.single_pass is True
    assert plan.epochs == 1
    assert plan.as_dict()["master_fineweb_mount_required"] is False
    # Pin identity unchanged vs prod window.
    assert plan.pin.identity_tuple() == prod_data_window().identity_tuple()


def test_offset_skips_leading_tokens() -> None:
    # 3 docs × 4 words each = 12 tokens total.
    docs = _docs("a b c d", "e f g h", "i j k l")
    # Skip first 5 tokens → start mid doc0 at index 1, take remaining toward budget 6.
    slices = list(
        apply_offset_and_budget(
            docs,
            token_start_offset=5,
            token_budget=6,
            encode_fn=_encode_identity_words,
        )
    )
    # d0 has 4 tokens; skip 5 means skip all of d0 (4) + 1 of d1 → start at d1 index 1.
    # Budget 6: d1 contributes 3 (indices 1..3), d2 contributes 3 → total 6.
    assert len(slices) == 2
    assert slices[0].document["id"] == "d1"
    assert slices[0].token_start == 1
    assert slices[0].token_count == 3
    assert slices[1].document["id"] == "d2"
    assert slices[1].token_start == 0
    assert slices[1].token_count == 3


def test_budget_stops_before_stream_end() -> None:
    docs = _docs("one two three four five", "six seven eight nine ten", "eleven twelve")
    # 5 + 5 + 2 words; budget 7 should stop mid second doc.
    slices = list(
        apply_offset_and_budget(
            docs,
            token_start_offset=0,
            token_budget=7,
            encode_fn=_encode_identity_words,
        )
    )
    total = sum(s.token_count for s in slices)
    assert total == 7
    assert slices[0].token_count == 5
    assert slices[1].token_count == 2
    assert slices[1].token_start == 0
    # Third doc never seen (single-pass stop).
    assert all(s.document["id"] != "d2" for s in slices)


def test_iter_tokens_respects_offset_and_budget() -> None:
    # Use sequential unique token ids to verify slice correctness.
    def enc(text: str) -> list[int]:
        # "t10 t11 t12" -> [10, 11, 12]
        return [int(w[1:]) for w in text.split()]

    docs = _docs("t0 t1 t2 t3 t4", "t5 t6 t7 t8 t9", "t10 t11 t12")
    tokens = list(
        iter_budgeted_tokens(
            docs,
            token_start_offset=3,
            token_budget=4,
            encode_fn=enc,
        )
    )
    # Skip t0,t1,t2 → start at t3; take t3,t4,t5,t6
    assert tokens == [3, 4, 5, 6]


def test_loader_with_mocked_source_stops_at_budget() -> None:
    # 10 docs of 10 words = 100 tokens available.
    docs = [{"id": str(i), "text": " ".join(f"w{i}x{j}" for j in range(10))} for i in range(10)]
    loader = EgalitarianFineWebLoader(
        document_source=docs,
        encode_fn=default_encode,
        token_budget=25,
    )
    plan = loader.plan()
    assert plan.token_budget == 25
    assert plan.pin.token_start_offset == EQUAL_OFFSET
    assert plan.single_pass is True
    n = loader.collect_token_count()
    assert n == 25


def test_loader_offset_plus_budget_on_tiny_stream() -> None:
    docs = [{"id": str(i), "text": "aa bb cc dd"} for i in range(20)]  # 4 tokens each
    pin = DataWindowPin(
        dataset_id=PROD_DATASET_ID,
        dataset_revision=PROD_DATASET_REVISION,
        token_start_offset=6,  # algorithm check: non-zero offset + budget stop
        token_budget=TOKEN_BUDGET_PROD,
        epochs=1,
    )
    loader = build_loader(
        pin,
        document_source=docs,
        encode_fn=default_encode,
        token_budget=10,
    )
    assert loader.plan().pin.token_start_offset == 6
    assert loader.plan().token_budget == 10
    assert loader.collect_token_count() == 10
    texts = list(loader.iter_text_shards())
    assert len(texts) >= 1
    # Re-iterating uses a fresh scan of the source list; still budget-capped (no multi-epoch
    # inflation beyond the single-pass window size).
    assert loader.collect_token_count() == 10


def test_prod_equal_offset_is_shared_identity() -> None:
    """Prod pin uses EQUAL_OFFSET for every architecture (identity tuple)."""
    a = prod_data_window()
    b = prod_data_window()
    assert a.token_start_offset == b.token_start_offset == EQUAL_OFFSET
    assert a.identity_tuple() == b.identity_tuple()


def test_single_pass_rejects_multi_epoch() -> None:
    docs = _docs("a b c")
    with pytest.raises(LoaderError, match="single-pass"):
        list(
            apply_offset_and_budget(
                docs,
                token_start_offset=0,
                token_budget=10,
                epochs=2,
            )
        )


def test_pin_validate_rejects_multi_epoch() -> None:
    with pytest.raises(ValueError, match="single-pass"):
        DataWindowPin(
            dataset_id=PROD_DATASET_ID,
            dataset_revision=PROD_DATASET_REVISION,
            token_start_offset=0,
            token_budget=100,
            epochs=3,
        ).validate()


def test_stream_exhaustion_without_rescan() -> None:
    """If the stream ends before budget, stop — never wrap for a second epoch."""
    docs = _docs("one two", "three four")
    tokens = list(
        iter_budgeted_tokens(
            docs,
            token_start_offset=0,
            token_budget=10_000,
            encode_fn=default_encode,
        )
    )
    assert len(tokens) == 4  # only what the single pass offered


def test_count_tokens_default_encode() -> None:
    assert count_tokens("hello world again") == 3
    assert count_tokens("") == 0


def test_plan_documents_no_master_mount() -> None:
    meta = build_loader().plan().as_dict()
    assert meta["master_fineweb_mount_required"] is False
    assert meta["token_budget_prod"] == 2_500_000_000
    assert meta["equal_offset"] == EQUAL_OFFSET
    assert meta["single_pass"] is True


def test_default_encode_stable_across_pythonhashseed() -> None:
    """default_encode must not depend on PYTHONHASHSEED (no builtin hash()).

    Given: the same multi-word input string
    When: encoded in fresh interpreters with PYTHONHASHSEED=0 and =1
    Then: both processes emit identical token id lists
    """
    import json
    import os
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """\
        import json
        from prism_recipe.loader import default_encode
        text = "alpha beta gamma hello world prism recipe stable"
        print(json.dumps(default_encode(text)))
        """
    )

    def _encode_under_seed(seed: str) -> list[int]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        # Prefer package on pythonpath from pytest config; keep src fallback.
        src = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src if not prev else f"{src}{os.pathsep}{prev}"
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    ids_seed_0 = _encode_under_seed("0")
    ids_seed_1 = _encode_under_seed("1")
    assert ids_seed_0 == ids_seed_1
    assert len(ids_seed_0) == 8
    assert all(isinstance(t, int) and t > 0 for t in ids_seed_0)


def test_default_encode_matches_fnv1a_smoke_path() -> None:
    """Synthetic ids follow the same FNV-1a path as smoke_train.fixture_encode."""
    from prism_recipe.smoke_train import fixture_encode

    text = "alpha beta gamma"
    # fixture_encode maps into vocab; default_encode uses full 31-bit positive space.
    # Both must be seed-stable and derived from the same per-word FNV-1a 32-bit hash.
    ids = default_encode(text)
    assert ids == default_encode(text)  # pure / repeatable in-process
    assert len(ids) == 3
    # Cross-check: fixture_encode uses FNV then % (vocab-1)+1; rebuild FNV here.
    def _fnv1a_32(word: str) -> int:
        h = 2166136261
        for ch in word.encode("utf-8"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    expected = [(_fnv1a_32(w) % (2**31 - 1)) or 1 for w in text.split()]
    assert ids == expected
    # Smoke path still produces positive in-vocab ids for the same words.
    smoke_ids = fixture_encode(text)
    assert len(smoke_ids) == 3
    assert all(1 <= t < 32000 for t in smoke_ids)  # MODEL_VOCAB_SIZE default path
