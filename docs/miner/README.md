# Miner guide — PRISM recipe

## What you deploy

Deploy the **prism-recipe** image. You do **not** submit a separate architecture ZIP for
this product path. The image already contains architecture, training, loader, rules, and
harness.

## Environment

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes (gate) | Pre-train LLM rules check |
| `PRISM_RECIPE_TOKEN_BUDGET` | No | Smoke / smaller budget override |
| Lium / Base worker env | Yes for live score | Host-side deploy (API key, master URL, hotkey binding) |

Never put keys in the Dockerfile, git, or logs.

## Deploy sketch

```bash
# After image is published and digest recorded (follow-on release):
#   IMAGE=ghcr.io/baseintelligence/prism-recipe@sha256:<digest>
#
# Host-side: Lium rent + worker agent enrollment bound to your miner hotkey.
# Inject OPENROUTER_API_KEY into the pod env only.

docker run --rm \
  -e OPENROUTER_API_KEY \
  prism-recipe:local \
  preflight
```

Live dual-worker score (two distinct miner-bound workers, peer attestor, full envelope) is
documented with the image release and worker-agent integration. Cap Lium spend and tear down
pods when finished.

## Egalitarian data

All architectures share:

- FineWeb-Edu dataset pin / revision
- Fixed global token start offset
- Production budget **2_500_000_000** tokens
- Single pass (one epoch)

Long 2.5B GPU runs are a production pin; engineering smoke uses tiny models and a small
`PRISM_RECIPE_TOKEN_BUDGET`.

## LLM gate

1. Image loads `.rules/*.md` and computes `rules_digest`.
2. OpenRouter call uses your key and a pinned model (client lands in a follow-on release).
3. Structured `{ok, reason, rules_digest, model, decision, checked_at}` always preferred over
   stack traces for expected rejects.
4. Missing key → structured fail; train does not start.
5. Gate metadata belongs on ExecutionProof / result payloads.

## Attestation checklist

- [ ] Digest-pinned image
- [ ] Gate metadata present (pass or fail path)
- [ ] Full worker envelope for Prism finalize (not stub-only)
- [ ] Peer path (no self-eval as sole scorer)
- [ ] Pods torn down; secrets not printed

## Local commands

```bash
pip install -e ".[dev]"
pytest -q
prism-recipe rules-digest
prism-recipe preflight
```
