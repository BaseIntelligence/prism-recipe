# PRISM recipe image — CPU smoke twin (CI / local digest pin).
# CUDA twin: Dockerfile.cuda (preferred for GPU miner deploys).
#
# Build + record digest:
#   docker build -t prism-recipe:local .
#   docker image inspect prism-recipe:local --format '{{index .RepoDigests 0}}'
#   # or for content digest after push / buildx:
#   docker buildx build --load -t prism-recipe:local . && \
#     docker image inspect prism-recipe:local --format '{{.Id}}'
#
# Image contents (sealed; no mid-run miner mutation path):
#   - harness (preflight / smoke train / train)
#   - tiny-1m architecture (≤1.5M params)
#   - .rules/ (LLM gate input)
#   - egalitarian FineWeb loader
#   - OpenRouter LLM gate

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

LABEL org.opencontainers.image.title="prism-recipe" \
      org.opencontainers.image.description="PRISM recipe harness: egalitarian HF window, OpenRouter gate, tiny-1m smoke train" \
      org.opencontainers.image.source="https://github.com/BaseIntelligence/prism-recipe" \
      org.opencontainers.image.licenses="Apache-2.0" \
      prism.recipe.variant="cpu-smoke" \
      prism.recipe.architecture="transformer-tiny-1m" \
      prism.recipe.mid_run_miner_mutation="false"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PRISM_RECIPE_HOME=/app \
    # Offline image smoke defaults (miner overrides in deploy).
    PRISM_RECIPE_SMOKE_SKIP_GATE=0 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY .rules ./.rules
COPY tests ./tests
COPY docs ./docs

# CPU torch wheel for offline smoke train (CI twin of CUDA image).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -e ".[dev,train]"

# Immutable contents check: rules + harness modules must be present.
RUN test -f /app/.rules/training.md \
    && test -f /app/src/prism_recipe/harness.py \
    && test -f /app/src/prism_recipe/arch/tiny_1m.py \
    && test -f /app/src/prism_recipe/loader.py \
    && test -f /app/src/prism_recipe/llm_gate.py \
    && prism-recipe rules-digest \
    && prism-recipe smoke --skip-gate

# Miner injects secrets at runtime only — never bake keys:
#   OPENROUTER_API_KEY  (LLM gate)
# Host-side only (never in image):
#   LIUM_API_KEY, miner hotkey material, wallet paths

ENTRYPOINT ["prism-recipe"]
CMD ["preflight"]
