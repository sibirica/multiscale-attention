"""
KVCache: Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/kv_cache.py
"""

import torch


class KVCache:
    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        return_full_cache: bool = False,
    ) -> None:
        cache_shape = (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = [torch.zeros(cache_shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(cache_shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.cache_len = [0] * n_layers
        self.n_layers = n_layers
        self.layer = 0
        self.device = device
        self.return_full_cache = return_full_cache

    @property
    def size(self) -> int:
        return self.cache_len[self.layer]

    def set_layer(self, layer: int):
        self.layer = layer

    def reset(self):
        self.layer = 0
        self.cache_len = [0] * self.n_layers

    @torch.compiler.disable()
    def update(self, k_val: torch.Tensor, v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.
        """
        bsz, _, seq_len, _ = k_val.shape
        l = self.layer
        k_cache, v_cache = self.k_cache[l], self.v_cache[l]

        if bsz > k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {k_cache.shape[0]}"
                f", but found new key tensors with batch size {bsz}!"
            )

        # update cache length
        prev_len = self.cache_len[l]
        cur_len = prev_len + seq_len
        if cur_len > k_cache.shape[2]:
            raise ValueError(
                f"KV cache sequence overflow: got len {cur_len}, max supported is {k_cache.shape[2]}."
            )
        self.cache_len[l] = cur_len

        # update cache
        k_cache[:bsz, :, prev_len:cur_len] = k_val
        v_cache[:bsz, :, prev_len:cur_len] = v_val

        if self.return_full_cache:
            k_out = k_cache[:bsz]
            v_out = v_cache[:bsz]
        else:
            k_out = k_cache[:bsz, :, :cur_len]
            v_out = v_cache[:bsz, :, :cur_len]

        return k_out, v_out
