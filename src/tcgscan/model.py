from __future__ import annotations

import torch
from torchvision.models import ResNet18_Weights, resnet18


def create_model(class_count: int, pretrained: bool = True) -> torch.nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    return model


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

