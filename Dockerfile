# syntax=docker/dockerfile:1.7.1@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# PRISM recipe image — CPU smoke twin (CI / local digest pin).
# CUDA twin: Dockerfile.cuda (preferred for GPU miner deploys).
#
# Hermetic + reproducible build notes:
#   - Base image pinned by sha256 digest (not a floating tag).
#   - Deps installed from hash-locked requirements.lock (+ CPU torch lock).
#   - No editable (-e) installs; package installed as a built wheel/sdist.
#   - SOURCE_DATE_EPOCH fixed; no apt-get (ca-certificates already in base).
#   - Build network denied except pinned registries (docker.io / registry-1.docker.io
#     and other hosts in prism_recipe.build_context.ALLOWED_BUILD_REGISTRIES).
#     Orchestrators must not pass arbitrary --build-context registry mirrors.
#     Miner trees never supply FROM/base image or a Dockerfile (build-prep rejects).
#   - Miner code (if present) lives only under /app/miner; sealed surfaces COPY after.
#   - Build with rewrite-timestamp for byte-identical content digests:
#       docker buildx build --builder default --load \
#         --provenance=false --sbom=false \
#         --build-arg SOURCE_DATE_EPOCH=1704067200 \
#         --output 'type=docker,name=prism-recipe:local,rewrite-timestamp=true' .
#
# Build + record content digest:
#   ./scripts/repro-build-cpu.sh   # dual no-cache build; requires RESULT=MATCH

ARG SOURCE_DATE_EPOCH=1704067200
# Digest resolved 2026-07-26 via registry-1.docker.io (library/python:3.12-slim-bookworm).
ARG BASE_IMAGE=python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b
FROM ${BASE_IMAGE}

# Re-declare after FROM so ENV expansion is non-empty (pre-FROM ARG scope ends at FROM).
ARG SOURCE_DATE_EPOCH=1704067200

LABEL org.opencontainers.image.title="prism-recipe" \
      org.opencontainers.image.description="PRISM recipe harness: egalitarian HF window, OpenRouter gate, tiny-1m smoke train" \
      org.opencontainers.image.source="https://github.com/BaseIntelligence/prism-recipe" \
      org.opencontainers.image.licenses="Apache-2.0" \
      prism.recipe.variant="cpu-smoke" \
      prism.recipe.architecture="transformer-tiny-1m" \
      prism.recipe.mid_run_miner_mutation="false"

ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    TZ=UTC \
    LC_ALL=C \
    LANG=C \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PRISM_RECIPE_HOME=/app \
    PRISM_RECIPE_SMOKE_SKIP_GATE=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=0

# ca-certificates is already present on python:3.12-slim-bookworm — no apt-get
# (apt-get update is non-reproducible across wall-clock builds).

WORKDIR /app

# Single COPY keeps context→layer mapping simple for rewrite-timestamp.
# ORDER IS LOAD-BEARING: miner subtree first, then sealed recipe surfaces so a
# hostile context cannot overwrite harness / rules / loader / gate / arch / train.
COPY requirements.lock requirements-torch-cpu.lock pyproject.toml README.md LICENSE ./
COPY miner ./miner
COPY src ./src
COPY .rules ./.rules
COPY tests ./tests
COPY docs ./docs

# Hash-checked dependency install (no -e). One RUN so the layer ends after
# install + seal checks + full mtime clamp to SOURCE_DATE_EPOCH.
RUN python -m pip install --require-hashes -r requirements.lock \
    && python -m pip install --require-hashes -r requirements-torch-cpu.lock \
    && python -m pip install --no-build-isolation --no-deps . \
    && test -f /app/.rules/training.md \
    && test -f /app/src/prism_recipe/harness.py \
    && test -f /app/src/prism_recipe/arch/tiny_1m.py \
    && test -f /app/src/prism_recipe/loader.py \
    && test -f /app/src/prism_recipe/llm_gate.py \
    && prism-recipe rules-digest \
    && prism-recipe sealed-manifest bake --out /app/sealed_surface_manifest.json \
    && test -f /app/sealed_surface_manifest.json \
    && prism-recipe sealed-manifest verify \
    && prism-recipe smoke --skip-gate \
    && rm -rf /tmp/* /root/.cache /var/tmp/* \
    && find /usr/local /app -xdev -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

# Mech 6 — build-time per-run attestation secret (BuildKit secret mount).
# The mount at /run/secrets/attestation_secret is NOT stored in the layer;
# only the destination file is. Never ARG/ENV/LABEL the secret value.
# Root miners can extract /run/prism/attestation_hmac_key — raises cost, not trust.
# Build with: --secret id=attestation_secret,src=/path/to/secret-file
#   and --build-arg PRISM_ATTESTATION_INJECT_ID=<unique-non-secret-id> (cache bust).
# (see scripts/repro-build-cpu.sh). Unique per build; code rootfs MATCH excludes this path.
#
# PRISM_ATTESTATION_INJECT_ID is NOT the secret — it only busts BuildKit cache so
# two builds with different secret files cannot reuse a cached secret layer.
ARG PRISM_ATTESTATION_INJECT_ID=unset
RUN --mount=type=secret,id=attestation_secret,required=true \
    test -n "${PRISM_ATTESTATION_INJECT_ID}" \
    && test "${PRISM_ATTESTATION_INJECT_ID}" != "unset" \
    && mkdir -p /run/prism \
    && install -m 0400 /run/secrets/attestation_secret /run/prism/attestation_hmac_key \
    && test -s /run/prism/attestation_hmac_key \
    && touch -h -d "@${SOURCE_DATE_EPOCH}" /run/prism /run/prism/attestation_hmac_key

# Miner injects runtime secrets only — never bake keys:
#   OPENROUTER_API_KEY  (LLM gate)
# Host-side only (never in image):
#   LIUM_API_KEY, miner hotkey material, wallet paths
# Build-time only (BuildKit secret mount → file, not ENV):
#   attestation_secret → /run/prism/attestation_hmac_key (mode 0400)

# Sidecar listen mode (optional). Publish this port in the Lium template's
# internal_ports so BASE can dial POST /v1/sidecar/attest on the instance.
EXPOSE 8787

ENTRYPOINT ["prism-recipe"]
CMD ["preflight"]
