# Security

## Secrets

| Secret | Where it lives | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Miner worker env at runtime | Required for rules gate; never commit or log |
| `LIUM_API_KEY` | Miner **host** deploy env | Not baked into the recipe image |
| Hotkey private material | Miner wallet host only | Remote-sign patterns preferred; never in image logs |

See [`.env.example`](../.env.example) for names only.

## Image integrity

- Prefer digest-pinned pulls (`image@sha256:...`) for scored deploys.
- Bundled `.rules/` are read-only inputs to the gate; runtime mutation is a reject condition.
- Swapping harness entrypoints without a new digest breaks the pin story.

## Attestation expectations

Worker results on the scoring path should carry:

- `rules_digest`, OpenRouter model id, gate decision, `checked_at`
- Image digest / pin identity
- Full finalize envelope (`run_manifest`, `manifest_sha256`, `execution_proof`) when
  posting to Prism

Gate skip is not allowed on scoring paths.

## Residual risk

- Third-party OpenRouter and Hugging Face availability and policy filters.
- Cryptographically-anchored proofs bind digests and metadata; they do not eliminate all
  hardware, supply-chain, or operator compromise risk.
- CPU placeholder image is for scaffolding and smoke shapes; production train expects a
  CUDA-capable image once published.
