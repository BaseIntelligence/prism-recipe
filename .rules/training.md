# Training Compliance

Training code that runs inside the PRISM recipe image must follow these rules.
The pre-train LLM gate evaluates architecture and training sources against this
file (and sibling rules) before any train step starts.

## Accept

- From-scratch training with forced or recipe-controlled random initialization.
- Single-pass consumption of the egalitarian FineWeb-Edu window defined by the
  recipe pin (fixed dataset revision, fixed global token start offset, production
  `token_budget=2_500_000_000`, one epoch).
- Smoke runs that shrink `token_budget` via `PRISM_RECIPE_TOKEN_BUDGET` without
  changing the offset pin identity.
- Standard optimizer / schedule choices that do not require external pretrained
  checkpoints or network fetches mid-train.
- Logging loss and metrics that the harness recomputes or records itself.

## Reject

- Loading pretrained weights, adapters, or external checkpoints to seed the run.
- Multi-epoch rescans or reshuffles that break the single-pass egalitarian window.
- Changing the FineWeb-Edu revision, start offset, or prod budget identity to game
  the comparison across architectures.
- Fetching arbitrary internet data or private miner-hosted corpora during train.
- Disabling, skipping, or short-circuiting the OpenRouter rules gate.
- Mutating bundled `.rules/` content at runtime.

## Reason Codes

- `pretrained_weights`
- `multi_epoch_rescan`
- `window_identity_tamper`
- `external_data_fetch`
- `gate_bypass`
- `rules_mutation`
