# BEYOND NATURAL IMAGES: A Benchmark for Cross-Domain Image Reconstruction

## Project Overview

This project will investigate how well state-of-the-art deep learning models for image reconstruction generalize across different visual domains. By evaluating CNN-, Transformer-, and Diffusion-based architectures on super- resolution, denoising, and inpainting tasks across diverse image types—such as natural scenes, text, astronomical, and stylized images—we aim to identify consistent strengths, weaknesses, and failure patterns. The findings will provide insights into cross-domain robustness and guide the development of more adaptable, content-aware reconstruction approaches

 We compare three model families:

- **CNNs** – EDSR, ESRGAN, DnCNN  
- **Transformers** – SwinIR, Uformer  
- **Diffusion Models** – Stable Diffusion Upscaler, SR3

Tasks:
1. **Super-Resolution**
2. **Denoising**

Domains (initially):
- **Natural Images** (e.g., DIV2K)
- **Text / Documents** (e.g., TextZoom-like datasets)
- **Astronomy** (star/galaxy images)

We care about:
- Traditional metrics: **PSNR, SSIM, LPIPS**
- Domain metrics: **OCR accuracy** (text), **star/flux preservation** (astronomy)
- **Cross-Domain Drop (CDD)** – how much performance degrades outside natural images.

---

## Project Structure

```text
Beyond-Natural-Images/
├── src/
│   ├── __init__.py
│   ├── benchmark.py        # unified CLI to run any model on any image
│   └── utils/
│       ├── __init__.py
│       └── image_io.py     # image loading / saving helpers
├── notebooks/              # exploratory analysis & prototyping
├── data/
│   └── README.md           # notes on datasets / download scripts
├── results/
│   └── README.md           # notes on where outputs & logs are stored
├── environment.yml         # conda environment for reproducibility
└── README.md               # this file
