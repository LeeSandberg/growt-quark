"""Feature extraction from PyTorch models for Growt auditing."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: Optional[str] = None,
    max_samples: int = 5000,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors from a model layer.

    Args:
        model: PyTorch model.
        dataloader: DataLoader yielding (inputs, labels) tuples or bare inputs.
        layer_name: Dot-separated layer name. If None, auto-detects the
            penultimate layer.
        max_samples: Maximum samples to extract.
        device: Device to run on. Defaults to model's device.

    Returns:
        (features, labels) as numpy arrays.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    target_layer = _resolve_layer(model, layer_name)

    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    collected = 0

    hook_output: list[torch.Tensor] = []

    def hook_fn(
        _module: torch.nn.Module, _input: tuple, output: torch.Tensor,
    ) -> None:
        hook_output.clear()
        out = output[0] if isinstance(output, tuple) else output
        hook_output.append(out.detach())

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for batch in dataloader:
                if collected >= max_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    batch_labels = (
                        batch[1] if len(batch) > 1
                        else torch.zeros(inputs.shape[0], dtype=torch.long)
                    )
                else:
                    inputs = batch.to(device)
                    batch_labels = torch.zeros(inputs.shape[0], dtype=torch.long)

                model(inputs)

                if hook_output:
                    feat = hook_output[0]
                    if feat.dim() > 2:
                        feat = feat.mean(dim=list(range(2, feat.dim())))
                    features_list.append(feat.cpu())
                    labels_list.append(batch_labels)
                    collected += feat.shape[0]
    finally:
        handle.remove()

    all_features = torch.cat(features_list, dim=0)[:max_samples]
    all_labels = torch.cat(labels_list, dim=0)[:max_samples]

    return all_features.numpy(), all_labels.numpy()


def _resolve_layer(
    model: torch.nn.Module, layer_name: Optional[str],
) -> torch.nn.Module:
    """Find the target layer for feature extraction."""
    if layer_name:
        parts = layer_name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    children = list(model.children())
    if len(children) >= 2:
        return children[-2]

    for _name, module in reversed(list(model.named_modules())):
        if not isinstance(module, (torch.nn.Linear, torch.nn.Softmax, torch.nn.LogSoftmax)):
            if list(module.parameters()):
                return module

    raise ValueError(
        "Could not auto-detect penultimate layer. "
        "Please specify layer_name explicitly."
    )
