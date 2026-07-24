# Miner guide — PRISM recipe

## What you deploy

Deploy the **prism-recipe** image. You do **not** submit a separate architecture ZIP for
this product path. The image already contains architecture (transformer-tiny-1m), training
smoke/harness, egalitarian loader, rules, and the OpenRouter LLM gate.

There is **no mid-run miner mutation path**: rules and harness are sealed in the image
digest. Changing them requires a new build and a new digest pin.

## Image variants

| File | Use |
| --- | --- |
| `Dockerfile` | **CPU smoke twin** (CI / local digest, offline `prism-recipe smoke`) |
| `Dockerfile.cuda` | **CUDA** base (preferred for live GPU miner deploys) |

### Build and record digest

```bash
# CPU smoke twin (local CI)
docker build -t prism-recipe:local .
IMAGE_ID=$(docker image inspect prism-recipe:local --format '{{.Id}}')
echo "image_id=$IMAGE_ID"

# Optional CUDA variant
docker build -f Dockerfile.cuda -t prism-recipe:cuda .
docker image inspect prism-recipe:cuda --format '{{.Id}}'
```

Record `image_id` / registry content digest in your deploy notes and worker pin config.
After a registry push, prefer `@sha256:<content-digest>` over a moving tag.

Local evidence file (not committed): `.docs-evidence/IMAGE_DIGEST.md`.

## Environment

### Inside the container (recipe runtime)

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes (gate) | Pre-train LLM rules check via OpenRouter |
| `PRISM_RECIPE_TOKEN_BUDGET` | No | Smoke / smaller budget override |
| `PRISM_RECIPE_SMOKE` | No | Force offline tiny-1m smoke train path |
| `PRISM_RECIPE_SMOKE_SKIP_GATE` | No | CI-only local gate pass (never for scored path) |

### Host-side only (never bake into the image)

| Variable / setting | Required for live score | Purpose |
| --- | --- | --- |
| `LIUM_API_KEY` | Yes | Lium rent / teardown on the **host** |
| Master / Base URL | Yes | `https://chain.joinbase.ai` (worker plane + Prism) |
| Hotkey **binding** | Yes | Enroll worker agent bound to miner SS58 hotkey |
| Wallet / remote-sign | Yes | Sign on dedicated host; private key never leaves host |

Never put keys in the Dockerfile, git, image layers, or logs.

## Deploy sketch

```bash
# After image is published and digest recorded:
#   IMAGE=ghcr.io/baseintelligence/prism-recipe@sha256:<digest>
#
# Host-side:
#   1) export LIUM_API_KEY (host env only)
#   2) rent GPU pod with IMAGE digest pin
#   3) enroll worker agent bound to miner hotkey (binding → master URL)
#   4) inject OPENROUTER_API_KEY into the pod env only
#   5) tear down pods when finished (cap spend)

docker run --rm \
  -e OPENROUTER_API_KEY \
  prism-recipe:local \
  preflight

# Offline tiny-1m smoke (no OpenRouter, fixture data)
docker run --rm \
  -e PRISM_RECIPE_SMOKE=1 \
  -e PRISM_RECIPE_SMOKE_SKIP_GATE=1 \
  -e PRISM_RECIPE_TOKEN_BUDGET=256 \
  prism-recipe:local \
  smoke --skip-gate
```

Live dual-worker score (two distinct miner-bound workers, peer attestor, full envelope) is
a follow-on feature. Cap Lium spend and tear down pods when finished.

## Egalitarian data

All architectures share the same window (no per-arch offsets):

| Pin | Value |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Revision (commit) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| `EQUAL_OFFSET` (global token start) | `0` |
| `TOKEN_BUDGET_PROD` | `2_500_000_000` |
| Epochs | `1` (single-pass; no multi-epoch rescan) |
| Smoke override | `PRISM_RECIPE_TOKEN_BUDGET` (budget only) |

Long 2.5B GPU runs are a **production config pin**, not a required live long train for
engineering smoke. Smoke uses **transformer-tiny-1m** (≤~1.5M params) and a small budget.

**Master FineWeb mount is not required.** Recipe workers stream HF FineWeb-Edu (or use
a local HF cache) themselves inside the image. Offline smoke uses a tiny fixture stream.

## Sealed architecture

| Item | Value |
| --- | --- |
| Name | `transformer-tiny-1m` |
| Source | `src/prism_recipe/arch/tiny_1m.py` (image-bundled) |
| Params | ~1.05M (cap 1.5M) |
| Smoke CLI | `prism-recipe smoke --skip-gate` |

## LLM gate

1. Image loads `.rules/*.md` and computes `rules_digest`.
2. OpenRouter call uses your key and a pinned model.
3. Structured `{ok, reason, rules_digest, model, decision, checked_at}` always preferred over
   stack traces for expected rejects.
4. Missing key → structured fail; train does not start.
5. Gate metadata belongs on ExecutionProof / result payloads.
6. Scored path must **not** set `PRISM_RECIPE_SMOKE_SKIP_GATE`.

## Attestation checklist

- [ ] Digest-pinned image (`docker image inspect` Id or registry `@sha256:…`)
- [ ] Image contains rules + harness (no mid-run miner mutation)
- [ ] Gate metadata present (pass or fail path)
- [ ] Full worker envelope for Prism finalize (not stub-only)
- [ ] Peer path (no self-eval as sole scorer)
- [ ] Host has `LIUM_API_KEY`; container has `OPENROUTER_API_KEY`; master URL + binding set
- [ ] Pods torn down; secrets not printed

## Local commands

```bash
pip install -e ".[dev,train]"
# CPU torch if needed:
#   pip install --index-url https://download.pytorch.org/whl/cpu torch

pytest -q
prism-recipe rules-digest
prism-recipe smoke --skip-gate
prism-recipe preflight   # needs OPENROUTER_API_KEY unless SMOKE_SKIP_GATE=1
```
