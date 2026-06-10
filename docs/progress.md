# Progress

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
