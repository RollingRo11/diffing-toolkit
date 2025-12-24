"""
Monkey-patch for dictionary_learning.cache.ActivationCache to handle
nnsight's lazy/meta device models.
"""

import torch
from dictionary_learning.cache import ActivationCache

# Store reference to the original collect method
_original_collect = ActivationCache.collect


def _get_target_device(model):
    """
    Get the appropriate device when model.device returns 'meta'.
    This happens with nnsight's lazy loading before dispatch.
    """
    device = model.device
    if device is not None and hasattr(device, 'type') and device.type == 'meta':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    return device


def _patched_collect(
    texts,
    submodules,
    submodule_names,
    model,
    out_dir,
    *args,
    **kwargs
):
    """
    Patched collect method that handles meta device models.
    Temporarily overrides model's device property to return the correct device.
    """
    original_device = model.device

    # Check if device is meta (nnsight lazy loading)
    if original_device is not None and hasattr(original_device, 'type') and original_device.type == 'meta':
        target_device = _get_target_device(model)

        # Store original device property from the class
        model_class = type(model)
        original_device_property = model_class.device

        # Temporarily override the device property
        model_class.device = property(lambda self: target_device)

        try:
            return _original_collect(
                texts, submodules, submodule_names, model, out_dir,
                *args, **kwargs
            )
        finally:
            # Restore the original device property
            model_class.device = original_device_property
    else:
        # Device is fine, just call the original
        return _original_collect(
            texts, submodules, submodule_names, model, out_dir,
            *args, **kwargs
        )


def apply_cache_patch():
    """Apply the monkey-patch to ActivationCache.collect."""
    ActivationCache.collect = _patched_collect


# Auto-apply the patch when this module is imported
apply_cache_patch()
