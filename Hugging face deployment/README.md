---
title: LISS-IV Cloud Reconstruction
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🛰️ TerraClear — Generative AI Cloud Removal for LISS-IV Satellite Imagery

**Bharatiya Antariksh Hackathon 2026 — Challenge 2**
*Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery*

Built for ISRO's Bharatiya Antariksh Hackathon 2026 (powered by Hack2skill). TerraClear reconstructs cloud-obscured regions in LISS-IV multispectral satellite tiles using a Partial-Convolution U-Net trained adversarially with a PatchGAN discriminator and VGG perceptual loss.

---

## The Problem

LISS-IV (Linear Imaging Self-Scanning Sensor-IV), onboard ISRO's Resourcesat-2/2A, captures high-resolution (~5.8 m) optical imagery across Green, Red, and NIR bands. Like all optical sensors, it cannot see through cloud cover — during India's monsoon season, when demand for fresh crop and disaster-monitoring imagery is highest, cloud contamination is also at its worst.

TerraClear addresses this by training a generative model to infer plausible, sharp ground-truth reconstructions underneath cloud masks, rather than defaulting to a blurry pixel-average of the training set (the classic failure mode of naive L1-loss inpainting).

---

## Architecture

| Component | Role |
|---|---|
| **Generator** — Partial U-Net (4-level encoder/decoder) | Uses Partial Convolutions so masked (cloud) pixels don't corrupt valid feature statistics; a learned "unknown region" embedding replaces the degenerate epsilon-collapse fallback under dense cloud (>70% coverage). |
| **Discriminator** — 70×70 PatchGAN | Penalizes locally flat/averaged output, forcing the generator to commit to sharp, textured predictions instead of a smoothed average. |
| **Perceptual loss** — VGG16 (ImageNet, first 16 layers) | Matches texture statistics of the prediction to ground truth, not just raw pixel values. |
| **Reconstruction loss** — Hole-weighted Charbonnier | Weights error inside cloud regions 6× higher than in visible regions; smoother gradients than plain L1 when combined with adversarial loss. |
| **Curriculum masking** | Epochs 1–15 train on small/medium (10–35%) cloud cover only; epochs 16+ introduce the full range up to 75% cover — prevents the model from learning "give up and average" as its first strategy. |

**Target metric:** SSIM 0.95–0.98, achievable in ~2–3 hrs / 60 epochs on a Colab T4 with 500–1000 training patches.

---

## App

A single Streamlit app (`app.py`) loads the ONNX model in-process and serves two modes:

- **📤 Upload your own image** — GeoTIFF (preferred, real multi-band), PNG/JPG, or a raw `.npy` training patch. Runs tiled inference (256×256 tiles, 32px overlap, feather-blended) so it works on scenes larger than one tile. No ground truth available for real uploads, so this mode is qualitative only.
- **🎲 Random demo patch** — for evaluators without their own satellite imagery. Picks a random known-clear `.npy` patch from `demo_patches/`, paints a realistic synthetic cloud onto it, reconstructs it, and reports **PSNR/SSIM against the real ground truth** — including PSNR restricted to just the cloud region, which is the number that actually reflects what the model had to invent rather than pixels it simply passed through.

Earlier versions of this project split the backend (FastAPI + ONNX Runtime) and frontend (Streamlit) into two separate services communicating over HTTP. They've since been merged into one process — the model loads once via `onnxruntime` directly inside Streamlit, removing the HTTP hop, CORS/XSRF configuration, and the "API not reachable" failure mode that came with running two services in one container.

---

## Model I/O (ONNX)

| Tensor | Shape | Notes |
|---|---|---|
| `cloudy_image` (input) | `[batch, 4, height, width]` | 4-channel patch (LISS-IV G/R/NIR + padding channel — see note below) |
| `cloud_mask` (input) | `[batch, 1, height, width]` | Binary cloud mask, 1 = cloud |
| `reconstructed_image` (output) | `[batch, 4, height, width]` | Reconstructed 4-channel output; RGB preview uses first 3 channels |

> **Note:** LISS-IV multispectral mode captures only 3 bands (Green, Red, NIR — no Blue). The 4th input/output channel is handled by repeating the NIR band rather than faking it from RGB, unless a real 4th band is available. See `app.py` → `build_model_inputs()`.

---

## Repository Structure

```
.
├── Dockerfile              # Single service — Streamlit only, model loaded in-process
├── README.md               # This file (also configures the HF Space, see YAML header above)
├── requirements.txt
├── app.py                  # Streamlit app — upload mode + synthetic-cloud demo mode with metrics
├── model.onnx              # Exported trained generator (Partial U-Net)
├── demo_patches/           # Known-clear .npy patches used by demo mode (add your own before deploying)
└── train_v2.py             # Full GAN training script (PyTorch) — not needed for inference/deployment
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running with Docker

```bash
docker build -t terraclear .
docker run -p 7860:7860 terraclear
```

## Deploying on Hugging Face Spaces

This repo is pre-configured as a **Docker Space** (see YAML header above). Upload:
`Dockerfile`, `README.md`, `app.py`, `requirements.txt`, `model.onnx`, and a `demo_patches/` folder containing a handful of your own known-clear `.npy` patches (4×H×W, normalized to [0,1]).
`model.onnx` (~30 MB) will be auto-tracked via Git LFS by the HF web uploader; if pushing via CLI, run `git lfs track "*.onnx"` first.

---

## Training

```bash
python train_v2.py --epochs 60 --amp          # fresh run
python train_v2.py --epochs 60 --amp --resume # resume from checkpoints/last_model.pth
```

Requires `.npy` training patches (4-channel, 256×256, values in [0,1]) under `data/patches/`, or a `--zip` archive of the same. Checkpoints and sample reconstructions are saved to `checkpoints/` and `logs/samples/` respectively.

---

## Known Limitations

- Real deployment accuracy depends on whether the uploaded image provides an actual NIR band (via GeoTIFF) — plain RGB uploads (PNG/JPG) will fall back to reduced accuracy.
- Very large scenes are processed via tiled inference — extremely large scenes will increase inference latency roughly linearly with tile count.
- Demo-mode metrics are computed against a *synthetic* cloud, not a real one — useful as a controlled sanity check, but not identical to accuracy on genuine LISS-IV cloud cover.
- VGG16 perceptual loss requires downloading pretrained ImageNet weights on first training run — no offline fallback by design (fails loudly rather than silently disabling the loss term).

---

## Team

*Built by Amrit Noor Singh and Kunal for ISRO Bharatiya Antariksh Hackathon 2026 — Challenge 2.*