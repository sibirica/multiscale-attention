"""Activation checkpointing wrapper for model submodules."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def checkpoint_wrapper(module: nn.Module) -> nn.Module:
    """Wrap a module forward with activation checkpointing during training."""
    if getattr(module, "_activation_checkpoint_wrapped", False):
        return module

    original_forward = module.forward

    def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
        if not module.training or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)

        def forward(*args: Any) -> Any:
            return original_forward(*args, **kwargs)

        return checkpoint(forward, *args, use_reentrant=False)

    module.forward = checkpointed_forward
    module._activation_checkpoint_wrapped = True
    return module
