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
│   ├── benchmark.py              # unified CLI to run any model on any image
│   ├── models/                   # baseline + real SR/Denoising models
│   │   ├── __init__.py
│   │   └── basic_models.py
│   └── utils/
│       ├── __init__.py
│       └── image_io.py           # image loading / preprocessing helpers
│
├── data/                         # datasets (NOT stored in GitHub)
│   ├── natural/
│   │   ├── div2k/
│   │   │   ├── HR/
│   │   │   └── LR_bicubic/
│   │   └── bsd/
│   │       ├── BSD68/
│   │       └── Set12/
│   │
│   ├── text/
│   │   ├── textzoom/
│   │   ├── synthetic_clean/
│   │   └── synthetic_noisy/
│   │
│   ├── astronomy/
│   │   ├── hubble/
│   │   └── deepsky/
│   │
│   ├── download_scripts/         # dataset download scripts (HPC-friendly)
│   │   ├── download_div2k.py
│   │   ├── download_bsd68.py
│   │   ├── download_deepsky.py
│   │   ├── download_hubble.py
│   │   └── download_textzoom.py
│   │
│   ├── preprocess/               # scripts to generate LR, noisy text, cleanup
│   │   ├── generate_LR_div2k.py
│   │   ├── make_text_synthetic.py
│   │   └── astro_cleanup.py
│   │
│   └── README.md                 # dataset documentation
│
├── results/                      # saved outputs, logs, metrics
│   └── README.md
│
├── notebooks/                    # prototyping / EDA
│
├── environment.yml               # local environment (Mac: CPU)
│
└── README.md                     # project overview
