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
