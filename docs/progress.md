# Progress

## Multiscale KV Cache Cleanup

### Objective
Clean up multiscale BCAT generation so KV rollout uses prebuilt fixed masks for static compile and caches K/V for all four attentions per layer, while staying aligned with the baseline BCAT cache setup.

### Task List
- [x] Align MultiscaleBCAT cache allocation and return_full_cache behavior with BCAT.
- [x] Prebuild fixed generation masks and remove per-step incremental mask construction from compiled rollout.
- [x] Refactor multiscale KV rollout to cache fast self, fast-to-slow, slow self, and slow-to-fast K/V.
- [x] Add focused tests for mask slicing and dense-vs-KV rollout behavior.
- [x] Run targeted validation.

### Current Focus
Completed targeted validation for mask parity and dense-vs-KV rollout behavior.

### Notes & Blockers
Multiscale KV rollout now uses four cache slots per layer and one fixed dense generation mask dictionary. Slow stream caching writes only completed slow blocks; the zero fallback for prefixes shorter than the pooling rate is not cached as a real slow token. Validation passed with `/home/daniel/miniconda3/bin/python` because the base interpreter does not have PyTorch installed.

---

## Residual LN modes (pre / peri / KEEL)

### Objective
Replace boolean `peri_ln` with a ternary residual-norm mode: pre-LN (baseline), peri-LN (`x + LN(f(LN(x)))`), and KEEL (`LN(α·x + f(LN(x)))` with α defaulting to `2*n_layer`). Peri-LN is not equivalent to KEEL with α=1 (different LN placement).

### Task List
- [x] Add `ln_mode` + `keel_alpha` config; wire through BCAT / MultiscaleBCAT / causal.
- [x] Implement pre / peri / KEEL branches in `CustomTransformerEncoderLayer` (+ cache path).
- [x] Implement matching branches in `TwoScaleTransformerEncoderLayer` (forward + KV).
- [x] Update yaml comments; run compile + causality tests.

### Current Focus
Done.

### Notes & Blockers
- peri: `y = x + LN(f(LN(x)))`. KEEL: `y = LN(α·x + f(LN(x)))`. Not the same at α=1.
- Multiscale peri still normalizes each self/cross head separately before summing; KEEL LNs the residual sum once.
- Config is just `ln_mode` + `keel_alpha` (no resolver helper); constructors take those kwargs directly.
- Default `keel_alpha` in yaml is `2*n_layer` (e.g. 12 for n_layer=6); code fallback is also `2 * config.n_layer`.

---

## VAE Patch Embedder for BCAT

### Objective
Add a BCAT patch embedding option based on a 2D VAE-style encoder/decoder while keeping positional encoding in the BCAT wrapper for that path.

### Task List
- [x] Inspect BCAT, existing embedders, and model config shape contracts.
- [x] Implement `src/models/vae.py` with a configurable 2D encoder/decoder and compression ratios 2/4.
- [x] Wire the new embedder into model construction and move learnable space/time embeddings into `BCAT`.
- [x] Keep VAE latent grids structured and centralize flatten/unflatten logic in `BCAT`.
- [x] Update the BCAT model config with VAE options while preserving the current ConvEmbedder default.
- [x] Run focused syntax and shape checks.
- [x] Add VAE-style spatial attention and validate 128x128 compression ratios 8/16/32.
- [x] Move BCAT space-time positional embeddings into `STPositionalEmbedding`.

### Current Focus
Completed implementation and checks.

### Notes & Blockers
VAE encode/decode now use `(b, t, ph, pw, d)` latent grids and support 128x128 compression ratios 8/16/32 for 16x16, 8x8, and 4x4 token grids. Full CPU forward is blocked by the existing CUDNN-only attention context when CUDA is unavailable, but helper-level encode/decode checks passed for VAE and the BCAT-owned Conv patch path.

---

## Upgrade VAE Embedder to Hunyuan-based 2D VAE

### Objective
Replace the VAE encoder/decoder internals in `src/models/vae.py` with a 2D (frame-independent) port of the Hunyuan-Video-1.5 VAE (`src/models/autoencoder_kl_hunyuanvideo15.py`) for better spatial compression, while preserving the `VAEEmbedder.encode`/`decode` shape contract.

### Task List
- [x] Review reference `autoencoder_kl_hunyuanvideo15.py`, current `VAEEmbedder`, `bcat.py` call sites, and embedder config.
- [x] Confirm design decisions with user (latent dim handling + deterministic vs probabilistic).
- [x] Port Hunyuan blocks to 2D in `src/models/vae.py` (Conv2d, RMSNorm, AttnBlock, Up/Downsample, Resnet, Mid/Down/Up blocks, Encoder2D, Decoder2D).
- [x] Build encoder/decoder outside `VAEEmbedder`; derive `block_out_channels` from existing config.
- [x] Validate shapes for ratios 8/16/32, backward pass, and `ruff` checks.

### Current Focus
Implementation complete and validated.

### Notes & Blockers
Key decisions (per user): 2D frame-independent (input `(b t) d h w`, all temporal components removed); deterministic autoencoder (no Gaussian sampling/KL). Channel alignment to/from the transformer `dim` is done with explicit 1x1 conv projections inside each module (`proj_out` in encoder, `proj_in` in decoder): the encoder/decoder run natively at the deepest feature width (`block_out_channels[-1]`) with plain Hunyuan residual shortcuts at the bottleneck, then project to/from `dim`. (Earlier `channel_resample` generalized shortcut approach was replaced by this on user request.) Mid-block-only single-head full spatial attention (Hunyuan style), so config `attention_resolutions`/`attn_heads` are now unused (left in place to avoid churn). Gradient-checkpointing and tiling code intentionally omitted. Verified: shapes for cr=8/16/32 (16/8/4 token grids), forward+backward with finite grads, `ruff format`/`ruff check` pass.
