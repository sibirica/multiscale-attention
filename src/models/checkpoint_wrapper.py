from __future__ import annotations
import functools
import logging
from collections.abc import Callable
from typing import Any, Literal
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing_extensions import TypeVar
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None
log = logging.getLogger(__name__)
_default_offload_to_cpu = False
_evaluating = False
def set_default_offload_to_cpu(offload: bool):
    """
    Set the default offload_to_cpu value for checkpoint_wrapper.
    This is used to avoid passing the offload_to_cpu argument every time.
    """
    global _default_offload_to_cpu
    _default_offload_to_cpu = offload
    log.info(
        f"Setting default offload_to_cpu for checkpoint_wrapper to {_default_offload_to_cpu}"
    )
def set_evaluating(evaluating: bool):
    """
    Set whether we are evaluating the model.
    This is used to avoid applying checkpoint_wrapper during evaluation.
    """
    global _evaluating
    _evaluating = evaluating
    log.info(f"Setting evaluating mode to {_evaluating}")
class GradientCheckpointingOffloader:
    """Dummy class (initialized by Hydra) which primarily serves to
    set the default offload_to_cpu value for checkpoint_wrapper.
    """
    def __init__(self, *, offload_to_cpu: bool):
        set_default_offload_to_cpu(offload_to_cpu)
TModule = TypeVar("TModule", bound=nn.Module, infer_variance=True)
def checkpoint_wrapper(
    m: TModule,
    offload_to_cpu: bool | Literal["auto"] | None = False,
) -> TModule:
    """
    A friendlier wrapper for performing activation checkpointing.
    Compared to the PyTorch version, this version:
wraps an nn.Module, so that all subsequent calls will use checkpointing
handles keyword arguments in the forward
handles non-Tensor outputs from the forward
    Usage::
        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu="auto")
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    """
    global _evaluating
    if _evaluating:
        return m
    global _default_offload_to_cpu
    if offload_to_cpu in (None, "auto"):
        offload_to_cpu = _default_offload_to_cpu
    # should I check whether original_forward has already been set?
    assert not hasattr(m, "precheckpoint_forward"), (
        "checkpoint function has already been applied?"
    )
    m.precheckpoint_forward = m.forward
    m.forward = functools.partial(
        _checkpointed_forward,
        m.precheckpoint_forward,  # original_forward
        offload_to_cpu,
    )
    return m
def unwrap_checkpoint(m: torch.nn.Module):
    """
    unwrap a module and its children from checkpoint_wrapper
    """
    for module in m.modules():
        if hasattr(module, "precheckpoint_forward"):
            module.forward = module.precheckpoint_forward
            del module.precheckpoint_forward
        if hasattr(module, "old_deepcopy_method"):
            module.__deepcopy__ = module.old_deepcopy_method
            del module.old_deepcopy_method
    return m
def _checkpointed_forward(
    original_forward: Callable,
    offload_to_cpu: bool,
    *args,
    **kwargs,
):
    """
If we need CPU-offload, fall back to the old custom Function
      (behaviour unchanged).
Otherwise, use PyTorch's built-in checkpoint with
      `use_reentrant=False`, which avoids the duplicate-gradient
      bug with DeepSpeed ZeRO 2/3.
    """
    if offload_to_cpu:
        # -------- keep old behaviour (offload requires custom path) --------
        kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
        parent_ctx_dict = {"offload": True}
        return CheckpointFunction.apply(
            original_forward, parent_ctx_dict, kwarg_keys, *flat_args
        )
    # ---------------- new fast-path ----------------
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    non_tensors: dict[str, Any] = {}
    def run_fn(*flat_inputs):
        unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, flat_inputs)
        out = original_forward(*unpacked_args, **unpacked_kwargs)
        if isinstance(out, torch.Tensor):
            return out
        # split non-tensor outputs so checkpoint sees only tensors
        tensors, packed = split_non_tensors(out)
        non_tensors["packed"] = packed
        return tensors
    tensors_out = checkpoint.checkpoint(
        run_fn,
        *flat_args,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    return (
        tensors_out
        if "packed" not in non_tensors
        else unpack_non_tensors(tensors_out, non_tensors["packed"])
    )
def pack_kwargs(*args, **kwargs) -> tuple[list[str], list[Any]]:
    """
    Usage::
        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == [1, 2]
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return kwarg_keys, flat_args
def unpack_kwargs(
    kwarg_keys: list[str], flat_args: list[Any]
) -> tuple[list[Any], dict[str, Any]]:
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs
def split_non_tensors(
    mixed: torch.Tensor | tuple[Any, ...],
) -> tuple[tuple[torch.Tensor], dict[str, list[Any]]]:
    """
    Usage::
        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors = []
    packed_non_tensors = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors
def unpack_non_tensors(
    tensors: tuple[torch.Tensor],
    packed_non_tensors: dict[str, list[Any]],
) -> tuple[Any]:
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict)
    mixed = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list)
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)
class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.
    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling ``unpack_non_tensors``.
    """
    @staticmethod
    def forward(ctx, run_function, parent_ctx_dict, kwarg_keys, *args):
        if torch.is_grad_enabled():  # grad may be disabled, e.g., during validation
            checkpoint.check_backward_validity(args)
        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()
        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(
                x.to(torch.device("cpu"), non_blocking=True) for x in tensor_inputs
            )
        else:
            ctx.fwd_device, ctx.grad_requirements = None, None
        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs
        with torch.no_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)
        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs
            return outputs
    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )
        tensor_inputs: tuple = ctx.saved_tensors
        tensor_inputs = checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = [
                t.to(ctx.fwd_device[i], non_blocking=True)
                for i, t in enumerate(tensor_inputs)
            ]
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)
        # Store the current states.
        bwd_rng_state = get_rng_state()
        # Set the states to what it used to be before the forward pass.
        set_rng_state(ctx.fwd_rng_state)
        with torch.enable_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)
        # Set the states back to what it was at the start of this function.
        set_rng_state(bwd_rng_state)
        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True, "
                "this checkpoint() is not necessary"
            )
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )
        return (None, None, None) + grads
def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if xm is not None:
        state["xla_rng_state"] = xm.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state
def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if xm is not None:
        xm.set_rng_state(state["xla_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])