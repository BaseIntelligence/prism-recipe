#!/usr/bin/env bash
# Dual no-cache CPU image build; compare content-addressed digests.
# Exit 0 only on RESULT=MATCH.
# Hermetic method (proven MATCH):
#   buildx --provenance=false --sbom=false
#   --output type=docker,...,rewrite-timestamp=true
#   SOURCE_DATE_EPOCH=1704067200
# Mech 6: each build gets a unique BuildKit secret (attestation_secret) plus a
# non-secret PRISM_ATTESTATION_INJECT_ID cache-bust so secret layers never reuse.
# The baked key path is excluded from rootfs_content_digest so code MATCH holds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EPOCH="${SOURCE_DATE_EPOCH:-1704067200}"
export SOURCE_DATE_EPOCH="$EPOCH"
TAG_A="${REPRO_TAG_A:-prism-recipe:repro-a}"
TAG_B="${REPRO_TAG_B:-prism-recipe:repro-b}"
BUILDER="${REPRO_BUILDER:-repro-builder}"

rootfs_content_digest() {
  local img="$1"
  local cid
  cid="$(docker create "$img" true)"
  docker export "$cid" | python3 "${ROOT}/scripts/rootfs_content_digest.py"
  docker rm -f "$cid" >/dev/null
}

# Read baked hmac key from image without printing it (sha256 only for evidence).
image_secret_fingerprint() {
  local img="$1"
  docker run --rm --entrypoint /bin/sh "$img" -c \
    'test -f /run/prism/attestation_hmac_key && test -s /run/prism/attestation_hmac_key && sha256sum /run/prism/attestation_hmac_key | cut -d" " -f1'
}

build_one() {
  local tag="$1"
  local secret_file="$2"
  local inject_id="$3"
  docker buildx build \
    --builder "$BUILDER" \
    --no-cache \
    --pull=false \
    --provenance=false \
    --sbom=false \
    --build-arg "SOURCE_DATE_EPOCH=${EPOCH}" \
    --build-arg "PRISM_ATTESTATION_INJECT_ID=${inject_id}" \
    --secret "id=attestation_secret,src=${secret_file}" \
    --output "type=docker,name=${tag},rewrite-timestamp=true" \
    -f Dockerfile \
    .
}

ensure_builder() {
  if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDER" --driver docker-container --use >/dev/null
  fi
  docker buildx inspect --bootstrap "$BUILDER" >/dev/null
}

make_secret() {
  # 32 random bytes hex — unique per build; never echo the value.
  local f
  f="$(mktemp -t prism-attestation-secret.XXXXXX)"
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32 >"$f"
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' >"$f"
  fi
  chmod 600 "$f"
  printf '%s' "$f"
}

echo "=== repro-build-cpu EPOCH=${EPOCH} builder=${BUILDER} ==="
echo "COMPARE_CMD=docker export \$(docker create IMG true) | python3 scripts/rootfs_content_digest.py"
echo "BUILD_CMD=docker buildx build --builder ${BUILDER} --no-cache --provenance=false --sbom=false --build-arg SOURCE_DATE_EPOCH=${EPOCH} --build-arg PRISM_ATTESTATION_INJECT_ID=ID --secret id=attestation_secret,src=SECRET_FILE --output type=docker,name=TAG,rewrite-timestamp=true -f Dockerfile ."
echo "SECRET_NOTE=unique per build; path /run/prism/attestation_hmac_key excluded from rootfs digest"

ensure_builder

SECRET_A="$(make_secret)"
SECRET_B="$(make_secret)"
cleanup() {
  rm -f "$SECRET_A" "$SECRET_B"
}
trap cleanup EXIT

FP_FILE_A="$(sha256sum "$SECRET_A" | awk '{print $1}')"
FP_FILE_B="$(sha256sum "$SECRET_B" | awk '{print $1}')"
echo "SECRET_FILE_SHA256_A=${FP_FILE_A}"
echo "SECRET_FILE_SHA256_B=${FP_FILE_B}"
if [[ "$FP_FILE_A" == "$FP_FILE_B" ]]; then
  echo "ERROR: generated identical secrets for A and B" >&2
  exit 1
fi

# Non-secret random cache-bust IDs (must NOT be derived from the secret).
INJECT_A="repro-a-$(openssl rand -hex 16 2>/dev/null || head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
INJECT_B="repro-b-$(openssl rand -hex 16 2>/dev/null || head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
echo "PRISM_ATTESTATION_INJECT_ID_A=${INJECT_A}"
echo "PRISM_ATTESTATION_INJECT_ID_B=${INJECT_B}"

build_one "$TAG_A" "$SECRET_A" "$INJECT_A"
build_one "$TAG_B" "$SECRET_B" "$INJECT_B"

ID_A="$(docker image inspect "$TAG_A" --format '{{.Id}}')"
ID_B="$(docker image inspect "$TAG_B" --format '{{.Id}}')"
echo "IMAGE_ID_A=${ID_A}"
echo "IMAGE_ID_B=${ID_B}"

DIGEST_A="$(rootfs_content_digest "$TAG_A")"
DIGEST_B="$(rootfs_content_digest "$TAG_B")"
echo "ROOTFS_CONTENT_DIGEST_A=${DIGEST_A}"
echo "ROOTFS_CONTENT_DIGEST_B=${DIGEST_B}"

KEY_FP_A="$(image_secret_fingerprint "$TAG_A")"
KEY_FP_B="$(image_secret_fingerprint "$TAG_B")"
echo "IMAGE_SECRET_SHA256_A=${KEY_FP_A}"
echo "IMAGE_SECRET_SHA256_B=${KEY_FP_B}"
if [[ "$KEY_FP_A" == "$KEY_FP_B" ]]; then
  echo "ERROR: two builds carried the same baked attestation secret" >&2
  exit 1
fi
echo "SECRET_DIFFERS_ACROSS_BUILDS=true"

exec python3 "${ROOT}/scripts/compare_repro_digests.py" \
  --image-id-a "$ID_A" \
  --image-id-b "$ID_B" \
  --rootfs-a "$DIGEST_A" \
  --rootfs-b "$DIGEST_B"
