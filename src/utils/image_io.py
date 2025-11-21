from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_for_classification(image_path: str, img_size: int = 224):
    """
    Load an image and preprocess it for ImageNet-style classification.
    Returns:
        tensor: (1, 3, H, W)
        pil_img: original PIL image
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    pil_img = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(pil_img).unsqueeze(0)
    return tensor, pil_img

def load_image_raw(image_path: str, img_size: int | None = None):
    from torchvision import transforms

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    pil_img = Image.open(path).convert("RGB")

    transform_list = []
    if img_size is not None:
        transform_list.append(transforms.Resize((img_size, img_size)))
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor









'''

def load_image_raw(image_path: str, img_size: int | None = None):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    pil_img = Image.open(path).convert("RGB")

    transform_list = []
    if img_size is not None:
        transform_list.append(transforms.Resize((img_size, img_size)))
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor
'''