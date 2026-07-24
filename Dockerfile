# PRISM recipe image (placeholder).
# Full CUDA / train deps land with the image+tiny-1m feature.
# Digest pin: build, tag, and record digest before miner deploy.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

LABEL org.opencontainers.image.title="prism-recipe" \
      org.opencontainers.image.description="PRISM recipe harness: egalitarian HF window, OpenRouter gate, attested train" \
      org.opencontainers.image.source="https://github.com/BaseIntelligence/prism-recipe" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal for the scaffold (no CUDA toolkit yet).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY .rules ./.rules
COPY tests ./tests

RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

# Miner injects secrets at runtime only:
#   OPENROUTER_API_KEY
# Host/Lium deploy env is separate (never bake LIUM_API_KEY into the image).
ENV PRISM_RECIPE_HOME=/app

ENTRYPOINT ["prism-recipe"]
CMD ["preflight"]
