---
name: coding-style
description: Coding style guidelines and common pitfalls for this project
---

# Coding Style

This file contains the coding style guidelines for the project. In general, we follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Type Hints

All function parameters and return values must have type annotations.

```python
# Good
def forward(self, data: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    ...

# Bad
def forward(self, data, mode):
    ...
```

Use `|` for union types (Python 3.10+):
```python
def get_lora(self, name: str) -> LoRALayerWeights | None:
    ...
```

## Docstrings

### Classes

Every class must have a brief docstring explaining its purpose.

```python
class IconLitModule(BaseLitModule):
    """Lightning module for ICON training and evaluation."""
```

### Public Methods

Public methods must have a docstring. Use the Google-style format with `Args` and `Returns` sections when the method has non-trivial parameters.

```python
def from_local_checkpoint(
    cls,
    lora_dir: str,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> "LoRAModel":
    """Create a LoRAModel from a local checkpoint.

    Args:
        lora_dir: The local path that has lora data.
        device: Device where the lora model is loaded.
        dtype: dtype of the lora model weights.

    Returns:
        Loaded LoRA Model.
    """
```

For simple methods, a one-line docstring is sufficient:

```python
def get_lora(self, module_name: str) -> LoRALayerWeights | None:
    """Get LoRA for a given module by name."""
    return self.loras.get(module_name, None)
```

### Private Methods

Private methods (prefixed with `_`) do not require docstrings, but add one if the logic is non-obvious.

### When to Add Docstrings

- **New code**: Always follow the above rules.
- **Existing code**: Add docstrings when you modify a function. Do not do bulk docstring additions in unrelated PRs.

## General

- **No Chinese comments.** Code in this repository is released publicly.
- **Avoid `__init__.py` files.** This promotes code isolation, simplifies maintenance, and minimizes merge conflicts. Import directly from the module file instead.

## Pitfalls

### Never use `cfg.get()`

Do not use `cfg.get(key)` or `cfg.get(key, default)` on Hydra `DictConfig` objects unless absolutely necessary. Missing keys silently return `None` or the default, masking typos and omitted config entries. Access keys directly (`cfg.key`) so a missing key raises an error immediately.

### No default parameters with `*args` or `**kwargs`

Do not combine default parameter values with `**kwargs` in the same function signature. When calling such a function, a misspelled keyword argument is silently absorbed by `**kwargs` instead of raising a `TypeError`.

```python
# Bad — misspelled kwarg silently goes into **kwargs, learning_rate stays at default
def train(model: nn.Module, learning_rate: float = 1e-3, **kwargs) -> None: ...
train(model, learing_rate=1e-4)  # typo, no error raised!

# Good — no **kwargs, so the typo raises TypeError immediately
def train(model: nn.Module, learning_rate: float = 1e-3) -> None: ...
train(model, learing_rate=1e-4)  # TypeError: unexpected keyword argument
```

The exception is to use `*args` and `**kwargs` when the function is a wrapper for another function with default parameters, or try to hide a lot of unused parameters for clarity.