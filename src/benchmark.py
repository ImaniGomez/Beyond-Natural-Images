# src/benchmark.py

import argparse
import time
from pathlib import Path

import torch

from .utils.image_io import (
    load_image_for_classification,
    load_image_raw,
)
from .models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark script for classification/SR/denoising."
    )

    parser.add_argument(
        "--task",
        type=str,
        default="classify",
        choices=["classify", "sr", "denoise"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "bicubic_x4", "gaussian_denoise"],
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--save_output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # Load input image
    if args.task == "classify":
        x, _ = load_image_for_classification(args.image_path, img_size=args.img_size)
    else:
        x = load_image_raw(args.image_path)
    x = x.to(device)

    # Load model
    model = get_model(args.model, device)

    # Warm-up
    with torch.no_grad():
        _ = model(x)

    # Timed inference
    n_runs = 5
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            out = model(x)
    end = time.time()

    print(f"\n=== Benchmark ===")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Avg inference time: {(end - start)*1000/n_runs:.3f} ms")
    print(f"Output shape: {out.shape}")

    # Save output if needed
    if args.save_output and args.task != "classify":
        out_dir = Path("results") / args.task / args.model
        out_dir.mkdir(parents=True, exist_ok=True)

        from torchvision.transforms.functional import to_pil_image
        out_img = out.cpu().clamp(0, 1)[0]
        out_path = out_dir / (Path(args.image_path).stem + f"_{args.task}.png")
        to_pil_image(out_img).save(out_path)
        print("Saved to:", out_path)


if __name__ == "__main__":
    main()
