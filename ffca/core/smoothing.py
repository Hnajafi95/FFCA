"""Universal smooth-activation context manager.

FFCA's Non-linearity and Interaction dimensions need meaningful second
derivatives, but ReLU's second derivative is identically zero. We swap
ReLU and friends for Softplus inside the analysis pass, and restore on
exit.

KNOWN LIMITATION: this only catches activations declared as `nn.Module`
attributes (e.g. `self.relu = nn.ReLU()`). Activations called as
functionals inside a `forward` (`F.relu(x)`) cannot be swapped; the user
must convert to module-style for FFCA analysis. We print a warning when
we walk the module tree and find zero replaceable activations.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Iterator

import torch.nn as nn


# Activations we know how to smooth out
_REPLACEABLE = (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.GELU, nn.SiLU)

# MaxPool variants are not twice-differentiable; replace with the equivalent AvgPool.
_MAXPOOL_TYPES = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)


def _make_avg_pool(m: nn.Module) -> nn.Module:
    """Build an AvgPool with the same geometry as the given MaxPool."""
    cls = {nn.MaxPool1d: nn.AvgPool1d, nn.MaxPool2d: nn.AvgPool2d,
           nn.MaxPool3d: nn.AvgPool3d}[type(m)]
    return cls(kernel_size=m.kernel_size, stride=m.stride,
               padding=m.padding,
               ceil_mode=getattr(m, "ceil_mode", False))


def _restore_max_pool(m: nn.Module) -> nn.Module:
    cls = {nn.AvgPool1d: nn.MaxPool1d, nn.AvgPool2d: nn.MaxPool2d,
           nn.AvgPool3d: nn.MaxPool3d}[type(m)]
    return cls(kernel_size=m.kernel_size, stride=m.stride,
               padding=m.padding,
               ceil_mode=getattr(m, "ceil_mode", False))


def _walk_replace(module: nn.Module, replacement_factory,
                  match=_REPLACEABLE) -> int:
    """In-place replace any matching activation with `replacement_factory()`."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, match):
            setattr(module, name, replacement_factory())
            n += 1
        else:
            n += _walk_replace(child, replacement_factory, match)
    return n


def _walk_replace_pools(module: nn.Module, factory) -> int:
    """Like _walk_replace but the factory takes the original pool module
    so we can copy its geometry."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, _MAXPOOL_TYPES):
            setattr(module, name, factory(child))
            n += 1
        else:
            n += _walk_replace_pools(child, factory)
    return n


def _walk_restore_pools(module: nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)) \
                and getattr(child, "_ffca_was_max", False):
            setattr(module, name, _restore_max_pool(child))
            n += 1
        else:
            n += _walk_restore_pools(child)
    return n


@contextlib.contextmanager
def _math_attention():
    """Force PyTorch's math-kernel SDPA so second-order derivatives work.

    PyTorch auto-selects flash / mem-efficient backends whose backward is not
    twice-differentiable. FFCA's Hessian probes need higher-order autograd.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        # Older PyTorch — fall back to setting flags
        import torch as _t
        prev = {
            "flash": _t.backends.cuda.flash_sdp_enabled(),
            "mem": _t.backends.cuda.mem_efficient_sdp_enabled(),
            "math": _t.backends.cuda.math_sdp_enabled(),
        }
        _t.backends.cuda.enable_flash_sdp(False)
        _t.backends.cuda.enable_mem_efficient_sdp(False)
        _t.backends.cuda.enable_math_sdp(True)
        try:
            yield
        finally:
            _t.backends.cuda.enable_flash_sdp(prev["flash"])
            _t.backends.cuda.enable_mem_efficient_sdp(prev["mem"])
            _t.backends.cuda.enable_math_sdp(prev["math"])
        return
    with ctx:
        yield


@contextlib.contextmanager
def smooth(model: nn.Module, beta: float = 10.0) -> Iterator[nn.Module]:
    """Context manager: replace ReLU-family activations with Softplus(beta).

    Args:
        model: the live model — modified in place during the with-block.
        beta:  Softplus sharpness. Higher β → closer to ReLU (less smoothing,
               smaller Hessian); lower β → smoother but larger curvature.
               The FFCA paper uses β=10 as a default.

    On exit, all activations are restored to `nn.ReLU()` (we don't try to
    remember the original subtype — leaky/PReLU/GELU all collapse to ReLU
    on restore. That's fine for inference; users should not train inside
    this context.)
    """
    n_replaced = _walk_replace(model, lambda: nn.Softplus(beta=beta))

    def _make_avg(m):
        a = _make_avg_pool(m)
        a._ffca_was_max = True  # tag for restore
        return a

    n_pools = _walk_replace_pools(model, _make_avg)
    if n_replaced == 0 and n_pools == 0:
        warnings.warn(
            "ffca.smooth(): no replaceable activations or MaxPools found. "
            "If you use F.relu()/F.max_pool2d() (functional API) inside "
            ".forward(), switch to nn.ReLU()/nn.MaxPool2d() etc. so FFCA can "
            "swap them. Non-linearity / Interaction will be ~0 otherwise.",
            stacklevel=2,
        )
    with _math_attention():
        try:
            yield model
        finally:
            _walk_replace(model, nn.ReLU, match=(nn.Softplus,))
            _walk_restore_pools(model)


def n_replaceable_activations(model: nn.Module) -> int:
    """Count how many module-style activations FFCA will be able to smooth."""
    n = 0
    for m in model.modules():
        if isinstance(m, _REPLACEABLE):
            n += 1
    return n
