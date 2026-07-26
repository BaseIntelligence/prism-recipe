# Security — PRISM recipe trust model

## One-sentence model

The miner **rents and controls** the Lium pod (including root). Attestation is
**tamper-evidence**, not tamper-prevention. A root miner can defeat every check
below. There is **no TEE**, and `effective_tier` never exceeds **1**.

## Who controls what

| Surface | Who controls it | Implication |
| --- | --- | --- |
| Git repo + commit SHA | Miner chooses; BASE builds | Digest maps to BASE-built code for that commit |
| Recipe image layers (harness, rules, data-window pins, sidecar) | BASE build of registered commit | Sealed in the image digest |
| Miner code under `miner/` | Miner | Their architecture/training code only |
| Lium pod (OS, root, runtime) | **Miner** (renter) | Root can inspect secrets, swap processes, forge responses |
| Lium API key | Miner supplies; BASE holds encrypted for constation | Same account that owns the pod |
| Lium TDX / host infra quotes (if any) | Lium executor | Attests **provider infrastructure**, not the renter workload |
| Score row on Prism | Prism ingestion + `constation_ok` | Missing/failed bundle → **no score row at all** (P1) |

## What this is not

- **Not** tamper-prevention. Root on the pod can change anything at runtime.
- **Not** TEE-grade assurance. No TDX/SEV/CVM path elevates score or tier.
- **Not** independent third-party verification of the pod. Mechanism 2 is
  **same-account corroboration** only (miner key + miner pod).
- **Not** a runtime measurement from Lium of the container rootfs. The Lium API
  digest is **declared template/config**, not a live integrity measurement.
- **Not** a claim that miners "cannot change anything." They can. We make
  dishonest runs expensive and visible, then refuse the score when evidence fails.

## The six attestation mechanisms (honest limitations)

All six must pass for `constation_ok` (and thus tier 1 / score admission under
P1). Each has a real residual gap.

### 1. Nonce-bound attestation

BASE issues a per-run nonce bound to work unit, miner hotkey, and pod. The
in-image sidecar signs `(nonce, digest, pod_id, timestamp, sealed hashes, …)`.
Replay of an older honest attestation fails once the nonce is consumed.

**Limitation:** A root miner who still holds the build secret and can answer
live challenges can mint fresh signatures for a swapped workload. Nonce stops
replay, not live forgery.

### 2. Same-account corroboration (two-channel cross-check)

BASE reads the pod/template digest from the Lium API (using the miner-supplied
account key) and requires it to agree with the sidecar's reported image digest.
Mismatch fails the bundle.

**Limitation:** This is **never independent verification**. The miner owns the
API key and the pod. Agreement only means the two channels under the same
account do not contradict each other. Agreement alone never elevates tier if
any other mechanism fails.

### 3. Continuous constation

Attest at run start, run end, and randomized intervals within a gap budget.
Gaps, timeouts, or mid-run digest drift fail the bundle (no score).

**Limitation:** A sophisticated root adversary can keep the sidecar honest while
running different code elsewhere, or answer polls from a twin process. Polling
raises cost and shrinks the swap window; it does not seal the kernel.

### 4. BASE-built digest allowlist

Only digests BASE produced from a registered `(commit_sha, tree_sha, variant)`
are scoreable. Unknown, revoked, or commit-mismatched digests fail closed.

**Limitation:** Allowlist proves "BASE once built this digest for that commit."
It does not prove the pod is still running that digest after deploy. Later
mechanisms carry that burden (imperfectly).

### 5. In-image self-measurement (sealed surfaces)

At runtime the sidecar hashes sealed harness / rules / data-window files and
includes them in the signed payload. Mutating those paths without a new BASE
build diverges from the baked manifest.

**Limitation:** Measurement is performed by software on a miner-controlled host.
Root can patch the sidecar, replace files after hash, or lie about paths. This
is evidence against casual mutation, not a hardware root of trust.

### 6. Build-time per-build secret

BASE injects a unique secret via BuildKit (`--secret id=attestation_secret` →
`/run/prism/attestation_hmac_key`, mode `0400`). The sidecar must prove
possession when answering challenges. Never `ARG`/`ENV`/`LABEL`. Hermetic code
digests exclude that path so two builds of the same commit can differ in secret
while code rootfs still matches.

**Limitation:** A **root miner on the pod can extract** the key file. The secret
raises the cost of forging responses offline; it does not create a hardware
root of trust.

## IMAGE_PIN, constation, and scoring (P1)

Historical docs called "IMAGE_PIN" a soft integrity label. Under the current
policy:

- Elevation and score admission require **`constation_ok`** over a full bundle
  (all six mechanisms), not a self-reported `PRISM_IMAGE_DIGEST` string match.
- `pinned_image_digest` may appear for telemetry; it **never** elevates tier alone.
- **Failed or missing attestation ⇒ no score row at all** (fail-closed).
- That gate is **decisive for score presence**, not orthogonal to ranking.
- `effective_tier` ceiling is **1**. Claimed tier ≥ 2 collapses to 0. No TEE path.

See Prism [security](../../prism/docs/security.md) and
[scoring](../../prism/docs/scoring.md).

## Secrets

| Secret | Where it lives | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Miner worker env at runtime | Required for rules gate; never commit or log |
| `LIUM_API_KEY` | Miner host + BASE constation custody (encrypted) | Full account power including pod destroy; never bake into image |
| Hotkey private material | Miner wallet host only | Remote-sign preferred; never in image logs |
| Build-time attestation secret (mech 6) | BuildKit secret → `/run/prism/attestation_hmac_key` | Unique per BASE build; root can extract on pod |

See [`.env.example`](../.env.example) for runtime names only.

## Image integrity (operational)

- Prefer digest-pinned pulls (`image@sha256:...`) for scored deploys.
- Bundled `.rules/` and sealed harness paths are read-only inputs; runtime
  mutation is a reject condition when measurement works.
- Swapping harness entrypoints without a new BASE build breaks the pin/allowlist
  story when constation is honest.
- CPU placeholder images are for scaffolding; production train expects a
  CUDA-capable image once published.

## Residual risk (explicit)

- Third-party OpenRouter and Hugging Face availability and policy filters.
- Cryptographic envelopes bind digests and metadata; they do not eliminate
  hardware, supply-chain, or operator compromise risk.
- A determined miner with root can defeat all six mechanisms. The design
  produces **evidence and cost**, not prevention.
- Closing the residual gap would require TEE and/or BASE-rented pods. Both are
  **out of scope** by product decision.
