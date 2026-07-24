# Security Policy

## Secrets

- Miners inject `OPENROUTER_API_KEY` (and Lium deploy credentials on the host)
  via environment only. Never bake keys into the image, git history, logs, or
  ExecutionProof payloads.
- Do not print API keys, wallet material, or private hotkey bytes.
- Prefer short-lived env injection on Lium pods; rotate keys if leaked.

## Isolation

- Training and gate logic run inside the digest-pinned recipe image.
- Network egress for OpenRouter is only for the pre-train rules gate (and HF
  dataset fetch / cache as documented by the loader). Arbitrary egress is not
  part of the product contract.
- Master coordination stays off the miner GPU path: master does not run PRISM
  GPU eval containers for this product.

## Residual Risk

- OpenRouter and Hugging Face are third-party services; availability and content
  filters are outside this repo's control.
- Cryptographically-anchored attestation binds digests and gate metadata; it
  does not claim absolute freedom from all hardware or supply-chain risk.
