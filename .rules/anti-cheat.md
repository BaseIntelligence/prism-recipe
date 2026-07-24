# Anti-Cheat Policy

Recipe evaluation must measure genuine architecture and training quality under a
shared data window. Reject shortcuts that hardcode scores, special-case the
harness, or escape the sealed image contract.

## Reject

- Reading, writing, or probing harness-internal score keys, expected loss tables,
  or peer attestor secrets to fabricate a better score.
- Branching on pod id, miner hotkey, work unit id, or environment labels only to
  change training quality on scored paths.
- Shipping static weights or logits that only satisfy tiny smoke fixtures.
- Disabling attestation fields (`rules_digest`, LLM decision, image digest) or
  posting empty ExecutionProof envelopes.
- Replacing the image-bundled loader / gate / train entrypoints at runtime while
  claiming the digest-pinned recipe image.

## Accept

- Honest from-scratch train under the sealed harness.
- Architecture or training experiments that stay within the image API contracts.
- Local smoke with smaller `PRISM_RECIPE_TOKEN_BUDGET` for engineering tests.

## Reason Codes

- `hardcoded_score`
- `harness_probe`
- `identity_branch`
- `attestation_strip`
- `image_swap`
