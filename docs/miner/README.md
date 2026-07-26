# Miner guide — PRISM recipe (constation path)

This document is enough to complete a **scored** submission on the miner-rent Lium
path. Read it end to end before deploying.

## Hard rules (read first)

1. You **rent and control** the Lium pod (including root). Attestation is
   **tamper-evidence**, not tamper-prevention. Root can defeat every check; when
   evidence fails, Prism simply **writes no score**.
2. **Failed or missing attestation = NO SCORE ROW AT ALL.** There is no
   partial credit, no "pin matched but constation soft-fail," and no tier from
   self-reported digests alone.
3. `effective_tier` never exceeds **1**. There is **no TEE** path and no
   independent third-party pod verifier.
4. Mechanism 2 (Lium API vs sidecar) is **same-account corroboration**, not
   independent verification. You supply the Lium API key for the same account
   that owns the pod.
5. Do **not** set `PRISM_RECIPE_SMOKE_SKIP_GATE` on any scored path.

Trust model detail: [../security.md](../security.md).

---

## Full submission flow

### 1. Submit git repo + commit SHA

- Put your architecture/training code under the recipe **`miner/`** tree only.
- Do **not** overwrite sealed recipe paths (harness, `.rules/`, data-window pins,
  Dockerfiles, attestation sidecar). Collisions are rejected at build assembly.
- Push a git repo and pin an exact **commit SHA** (and matching tree) that BASE
  will build. Moving tags are not a pin.

### 2. BASE builds reproducibly

- BASE (not you) builds the prism-recipe image for that commit with hermetic
  pins, sealed-surface manifest bake, and a **per-build attestation secret**
  (BuildKit secret → `/run/prism/attestation_hmac_key`).
- You do not inject the attestation secret yourself for scored builds.
- Local CPU smoke builds are for development only; scored digests come from BASE.

### 3. Digest allowlist registration

- After a successful BASE build, the content digest is registered on the
  allowlist for `(commit_sha, tree_sha, variant, digest)`.
- Unknown, revoked, or commit-mismatched digests are **unscoreable**.
- Prefer registry pulls as `image@sha256:<content-digest>` (not a floating tag).

### 4. Deploy to Lium

Host-side only (never bake into the image):

| Setting | Required | Purpose |
| --- | --- | --- |
| `LIUM_API_KEY` | Yes | Rent / teardown; also used by BASE constation custody |
| Master / Base URL | Yes | e.g. `https://chain.joinbase.ai` |
| Hotkey binding | Yes | Worker agent bound to miner SS58 hotkey |
| Wallet / remote-sign | Yes | Private key stays on signing host |

Pod runtime:

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes (gate) | Pre-train LLM rules check |
| Image digest pin | Yes | Deploy the **allowlisted** digest BASE built |

```bash
# Sketch only — use the BASE/worker enrollment path for live score.
# IMAGE=ghcr.io/.../prism-recipe@sha256:<allowlisted-digest>
# 1) export LIUM_API_KEY on the host
# 2) rent GPU pod with IMAGE digest pin
# 3) enroll worker agent bound to miner hotkey
# 4) inject OPENROUTER_API_KEY into pod env only
# 5) tear down pods when finished (cap spend)
```

### 5. Sidecar attests

The in-image sidecar must:

- Answer **nonce-bound** challenges from BASE (single-use, bound to work unit /
  hotkey / pod).
- Report image digest and **sealed-surface hashes** (harness / rules / data-window).
- Prove possession of the **build secret** without logging it.
- Answer **continuous** polls (start, randomized mid-run, end) within the gap budget.

### 6. Constation bundle → Prism ingest

BASE assembles a constation bundle (allowlist hit, nonce consume, signature,
sealed hashes, same-account corroboration, gap record) and forwards the worker
result envelope to Prism. Prism runs `constation_ok` as the sole elevation
predicate and the P1 score gate.

### 7. Score only if `constation_ok`

| Bundle state | Score row | Tier |
| --- | --- | --- |
| Valid, all six mechanisms pass | Written (then emission ranking applies) | Up to 1 |
| Missing | **None** — `miner_fault:missing_constation_bundle` | n/a |
| Any mechanism fails | **None** — `miner_fault:<code>` | n/a |
| Infra outage after retries | **None** — `infra_fault:*` (operator break-glass only) | 0 if admitted |

---

## Sealed vs miner-controlled

| Sealed (BASE image / recipe) | Miner-controlled |
| --- | --- |
| Harness entrypoints (`smoke_train` / `gpu_train`, etc.) | Code under `miner/` |
| `.rules/*` and rules digest inputs | Choice of commit SHA to submit |
| Data-window pins (dataset revision, egalitarian offsets) | Lium pod OS and root |
| Attestation sidecar + sealed-surface manifest | `LIUM_API_KEY`, wallet keys |
| Per-build attestation secret (injected by BASE) | `OPENROUTER_API_KEY` at runtime |
| Allowlisted image digest for that commit | Whether you keep the pod honest |

Changing sealed surfaces requires a **new BASE build** and a **new allowlist
entry**. Mid-run mutation is a reject condition when measurement works.

---

## `miner_fault:*` reason codes and remedies

These are the codes Prism ingestion / constation / break-glass emit for
miner-attributable failures. Each means **no score row**.

| Code | Meaning | Remedy |
| --- | --- | --- |
| `miner_fault:missing_constation_bundle` | No constation bundle on the result | Complete the full flow; ensure BASE constation ran and the envelope carries the bundle |
| `miner_fault:unknown_digest` | Image digest not on the allowlist | Deploy only the digest BASE built and registered for your commit |
| `miner_fault:variant_mismatch` | Allowlist variant (cpu/cuda/…) does not match | Deploy the variant that matches the registered allowlist row |
| `miner_fault:commit_mismatch` | Digest registered for a different commit/tree | Rebuild/register the commit you actually submitted; do not retag foreign digests |
| `miner_fault:revoked_digest` | Digest was revoked | Use a current non-revoked BASE build; contact ops if unexpected |
| `miner_fault:allowlist_failed` | Allowlist check failed (unmapped) | Re-check commit, tree, variant, digest registration; re-run BASE build path |
| `miner_fault:replayed_nonce` | Nonce already consumed | Do not reuse attestation responses; answer each challenge once |
| `miner_fault:unknown_nonce` | Nonce was never issued | Only answer nonces BASE issued for this work unit |
| `miner_fault:expired_nonce` | Nonce past TTL | Answer promptly; reduce pod/network delay; request a fresh work unit if needed |
| `miner_fault:nonce_work_unit_mismatch` | Nonce bound to another work unit | Keep pod/work-unit binding; do not mix envelopes across jobs |
| `miner_fault:nonce_hotkey_mismatch` | Nonce bound to another hotkey | Enroll and sign with the same hotkey BASE bound |
| `miner_fault:nonce_pod_mismatch` | Nonce bound to another pod | Do not move nonces across pods; re-issue on the active pod |
| `miner_fault:nonce_failed` | Nonce check failed (unmapped) | Re-pull work unit; ensure sidecar clock/binding fields match BASE |
| `miner_fault:signature_invalid` | Attestation signature / payload verify failed | Keep build secret intact; do not alter payload fields; use the image BASE built |
| `miner_fault:manifest_mismatch` | Sealed harness/rules/data-window hashes diverge | Do not mutate sealed paths; redeploy clean allowlisted image |
| `miner_fault:corroboration_mismatch` | Lium API declared digest ≠ sidecar digest | Pin the same digest in the Lium template and running container; no mid-run image swap |
| `miner_fault:constation_gap` | Poll gap exceeded budget | Keep sidecar up for the whole run; stable network; no long freezes |
| `miner_fault:unknown` | Fallback miner fault with empty/unknown bare code | Inspect full reject payload; fix underlying attestation; re-submit cleanly |

**Break-glass cannot admit `miner_fault` runs.** Only `infra_fault:*` may be
operator-overridden (audited, tier 0). Infra codes (for awareness, not miner
self-serve): `infra_fault:constation_unavailable`, `infra_fault:lium_5xx`,
`infra_fault:network_partition`, `infra_fault:constation_retry_exhausted`.

---

## Image variants (local dev)

| File | Use |
| --- | --- |
| `Dockerfile` | CPU smoke twin (CI / local digest) |
| `Dockerfile.cuda` | CUDA base (preferred for live GPU) |

```bash
# Local smoke only — scored digests come from BASE.
docker build -t prism-recipe:local .
docker image inspect prism-recipe:local --format '{{.Id}}'
```

## LLM gate (recipe)

1. Image loads `.rules/*.md` and computes `rules_digest`.
2. OpenRouter call uses your key and a pinned model.
3. Structured gate metadata on proofs; missing key → structured fail.
4. Scored path must **not** set `PRISM_RECIPE_SMOKE_SKIP_GATE`.

## Egalitarian data (sealed window)

| Pin | Value |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| `EQUAL_OFFSET` | `0` |
| `TOKEN_BUDGET_PROD` | `2_500_000_000` |
| Epochs | `1` (single-pass) |

## Local commands

```bash
pip install -e ".[dev,train]"
pytest -q
prism-recipe rules-digest
prism-recipe smoke --skip-gate   # local only
prism-recipe preflight           # needs OPENROUTER_API_KEY unless smoke skip
```

## Checklist before you expect a score

- [ ] Commit SHA registered; BASE build succeeded
- [ ] Digest on allowlist for that commit/tree/variant
- [ ] Lium pod runs that exact `@sha256:…` digest
- [ ] Sidecar answering nonces continuously; no gap over budget
- [ ] Sealed surfaces untouched; build secret present
- [ ] Lium template digest agrees with running image (same-account corroboration)
- [ ] Full worker envelope + constation bundle reached Prism
- [ ] No `miner_fault:*` on the reject path
- [ ] Pods torn down; secrets not printed

If any attestation step fails: **you get no score row**. Fix the code above and
re-run; do not expect ranking on a rejected ingest.
