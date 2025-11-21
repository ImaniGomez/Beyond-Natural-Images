# src/models/__init__.py

from .basic_models import (
    load_resnet18_classifier,
    BicubicSRx4,
    GaussianDenoise,
)

MODEL_REGISTRY = {
    "resnet18": load_resnet18_classifier,
    "bicubic_x4": BicubicSRx4,
    "gaussian_denoise": GaussianDenoise,
}

def get_model(name: str, device):
    """
    Returns a model/classifier based on name.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")

    constructor = MODEL_REGISTRY[name]

    if name == "resnet18":
        return constructor(device)
    else:
        return constructor()
