"""Offline tests for short GPU/CPU train fingerprint (VAL-GPU-H200-002 shape)."""

from __future__ import annotations

import math

from prism_recipe.gpu_train import build_gpu_run_manifest, run_gpu_short_train


def test_build_gpu_run_manifest_scoreable_shape() -> None:
    m = build_gpu_run_manifest(
        submission_id="t1",
        param_count=1_049_216,
        tokens_consumed=8_192,
        covered_bytes=8_192 * 6,
        online_loss=[4.0, 3.5, 3.0, 2.5],
        sum_neg_log_likelihood_nats=900.0,
        device="cuda:0",
        cuda_name="NVIDIA H200",
        token_budget=65536,
        steps=32,
        final_loss=2.5,
        wall_clock_seconds=12.0,
    )
    assert m["schema_version"] == "prism_run_manifest.v2"
    assert m["compute"]["model_params"] == 1_049_216
    assert m["metrics"]["tokens_consumed"] == 8_192
    assert m["metrics"]["predicted_tokens"] == 8_192
    assert m["data"]["covered_bytes"] > 0
    assert m["anti_cheat"]["step0_anomaly"] is False
    # score math mirrors score_prequential_bpb core
    bpb = (900.0 / math.log(2.0)) / m["data"]["covered_bytes"]
    assert 0 < bpb < 64


def test_run_gpu_short_train_cpu_fingerprint() -> None:
    r = run_gpu_short_train(
        submission_id="local",
        token_budget=4096,
        skip_gate=True,
        seq_len=64,
        batch_size=4,
        max_steps=16,
        require_cuda=False,
    )
    assert r.ok, r.message
    assert r.param_count >= 100_000
    assert r.tokens_consumed > 360
    assert r.run_manifest is not None
    assert r.run_manifest["compute"]["model_params"] == r.param_count
    assert r.param_count != 1088  # not TinyLM
