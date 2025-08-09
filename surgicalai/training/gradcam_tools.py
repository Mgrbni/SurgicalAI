# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

try:  # optional dependency
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception:  # pragma: no cover
    GradCAM = None  # type: ignore


def max_gradcam(model: torch.nn.Module, img: torch.Tensor) -> float:
    """Return max GradCAM value for a single image tensor."""
    if GradCAM is None:
        return 0.0
    try:
        target_layer = model.layer4[-1]
    except Exception:  # pragma: no cover - non-resnet
        target_layer = list(model.children())[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img, targets=[ClassifierOutputTarget(0)])
    return float(grayscale_cam.max())
