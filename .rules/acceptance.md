# Acceptance Policy

Accept a recipe run only when the sealed harness completes the required stages
with honest artifacts.

## Accept

- Pre-train LLM rules gate runs against image-bundled `.rules/` and records
  `rules_digest`, model id, decision, and `checked_at` in result metadata.
- Egalitarian FineWeb-Edu window identity is respected (revision + offset pin;
  prod budget 2.5B tokens, single pass).
- Train loop executes under the recipe image (architecture and training live in
  the image for this product path; no separate miner ZIP submit).
- Worker result includes a full envelope suitable for Prism finalize
  (`run_manifest`, `manifest_sha256`, `execution_proof` with provider and pod id
  when deployed on Lium).
- Smoke path may use tiny models and reduced `PRISM_RECIPE_TOKEN_BUDGET`.

## Reject

- Missing `OPENROUTER_API_KEY` presented as a crash instead of structured fail.
- Gate skip on any scoring path.
- Submissions that depend on a separate miner architecture ZIP as the primary
  path for this product (recipe image is the unit of deployment).
- Master-local GPU docker eval as the score path.
- Incomplete attestation metadata or stub executors that omit `run_manifest`.
