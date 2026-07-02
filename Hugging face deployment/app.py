"""
TerraClear — single-file Streamlit app for LISS-IV cloud reconstruction.

Merged from the previous FastAPI + Streamlit split into one process:
model loads once via onnxruntime directly inside Streamlit, no HTTP hop,
no CORS/XSRF flags needed, no "API not reachable" failure mode.

Two modes:
  1. Upload your own image (GeoTIFF / PNG / JPG / .npy) -> reconstruct, no ground truth.
  2. Random Demo Patch -> pick a real clear .npy patch from demo_patches/, paint a
     synthetic-but-realistic cloud on it, reconstruct, and score against the known
     ground truth with PSNR/SSIM. Useful when evaluators don't have their own imagery,
     and gives a real number instead of an unverifiable "looks fine" image.
"""
import glob
import io
import os
import random
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
import rasterio
import streamlit as st
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

st.set_page_config(page_title="TerraClear — LISS-IV Cloud Reconstruction", page_icon="🛰️", layout="wide")

MODEL_PATH = Path(__file__).parent / "model.onnx"
DEMO_DIR = Path(__file__).parent / "demo_patches"
TILE, OVERLAP = 256, 32


@st.cache_resource
def load_model():
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


session = load_model()


# ---------- Data loading (shared by both modes) ----------

def read_bands(file_bytes: bytes) -> np.ndarray:
    """Real multi-band GeoTIFF -> (bands, H, W) float32 in [0,1]."""
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        arr = src.read().astype(np.float32)
        arr = arr / max(arr.max(), 1.0)
    return arr


def read_npy(file_bytes: bytes) -> np.ndarray:
    """Load a training-pipeline .npy patch (C,H,W) or (H,W,C), normalized to [0,1]."""
    arr = np.load(io.BytesIO(file_bytes), allow_pickle=False).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr.shape}")
    if arr.shape[0] not in (4, 5) and arr.shape[-1] in (4, 5):
        arr = arr.transpose(2, 0, 1)
    if arr.max() > 1.5:
        arr = arr / max(arr.max(), 1.0)
    return arr


def build_model_inputs(bands: np.ndarray) -> np.ndarray:
    """Map real sensor bands -> the 4 channels the model expects.
    LISS-IV multispectral = Green, Red, NIR (3 bands, no Blue) -> pad by repeating NIR."""
    n = bands.shape[0]
    idx = [0, 1, 2, 3] if n >= 4 else [0, 1, 2, n - 1]
    return bands[idx]


def cloud_mask_from_bands(bands: np.ndarray, threshold: float) -> np.ndarray:
    """Clouds are bright across ALL bands -> use mean reflectance, not single-channel gray."""
    return (bands.mean(axis=0) > threshold).astype(np.float32)


# ---------- Synthetic cloud generation (ported exactly from train_v2.py) ----------
# IMPORTANT: this mirrors make_cloud_mask() + apply_cloud() from the training script,
# including the soft (non-binary) blurred mask. The model was trained on that soft
# mask as direct input -- feeding it a hard 0/1 mask instead is out-of-distribution
# and was producing the speckled/checkerboard artifacts seen in early demo testing.

def make_cloud_mask(size: int, frac_range=(0.10, 0.75), rng=None) -> np.ndarray:
    rng = rng or np.random
    mask = np.zeros((size, size), dtype=np.float32)
    scale = rng.uniform(30, 100)
    amp, freq = 1.0, 1.0
    for _ in range(6):
        layer = rng.standard_normal((size, size)).astype(np.float32)
        k = max(3, int(scale / freq))
        k = k if k % 2 == 1 else k + 1
        layer = cv2.GaussianBlur(layer, (k, k), 0)
        mask += amp * layer
        amp *= 0.5
        freq *= 2.0
    mn, mx = mask.min(), mask.max()
    if mx - mn < 1e-8:
        return np.zeros((size, size), dtype=np.float32)
    mask = (mask - mn) / (mx - mn)

    frac = rng.uniform(*frac_range)
    thresh = float(np.percentile(mask, (1 - frac) * 100))
    binary = (mask >= thresh).astype(np.float32)
    return cv2.GaussianBlur(binary, (15, 15), 0)


def apply_cloud(clean_hwc: np.ndarray, mask: np.ndarray, rng=None) -> np.ndarray:
    rng = rng or np.random
    H, W, C = clean_hwc.shape
    b = rng.uniform(0.72, 0.96)
    cloud = np.full((H, W, C), b, dtype=np.float32)
    cloud += rng.standard_normal((H, W, C)).astype(np.float32) * 0.03
    m3 = mask[:, :, None]
    return np.clip(cloud * m3 + clean_hwc * (1 - m3), 0.0, 1.0)


def add_synthetic_cloud(clean: np.ndarray, seed: int | None = None):
    """clean: (C,H,W) -> returns (cloudy (C,H,W), soft_mask (H,W)), using the
    exact same simulation the model was trained on."""
    rng = np.random.default_rng(seed)
    c, h, w = clean.shape
    mask = make_cloud_mask(h, rng=rng)
    clean_hwc = clean.transpose(1, 2, 0)
    cloudy_hwc = apply_cloud(clean_hwc, mask, rng=rng)
    return cloudy_hwc.transpose(2, 0, 1).astype(np.float32), mask.astype(np.float32)



# ---------- Inference (tiled, works for any scene size) ----------

def run_tiled_inference(cloudy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    c, h, w = cloudy.shape
    out = np.zeros((c, h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    step = TILE - OVERLAP
    for y in range(0, max(h - OVERLAP, 1), step):
        for x in range(0, max(w - OVERLAP, 1), step):
            y2, x2 = min(y + TILE, h), min(x + TILE, w)
            y, x = max(y2 - TILE, 0), max(x2 - TILE, 0)
            tile_in = cloudy[:, y:y + TILE, x:x + TILE][None].astype(np.float32)
            tile_mask = mask[y:y + TILE, x:x + TILE][None, None].astype(np.float32)
            result = session.run(
                ["reconstructed_image"], {"cloudy_image": tile_in, "cloud_mask": tile_mask}
            )[0][0]
            wy = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE))
            w2d = (np.outer(wy, wy).astype(np.float32))
            w2d /= w2d.max()
            out[:, y:y + TILE, x:x + TILE] += result * w2d
            weight[y:y + TILE, x:x + TILE] += w2d
    return out / np.clip(weight, 1e-6, None)[None]


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """(C,H,W) float [0,1] -> (H,W,3) uint8 for display."""
    rgb = arr[:3]
    return np.clip(rgb.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)


def compute_metrics(clean: np.ndarray, recon: np.ndarray, mask: np.ndarray):
    """PSNR/SSIM over the full patch, and PSNR restricted to the cloud region
    only -- the number that actually matters, since visible pixels are trivial
    to 'reconstruct' by just copying them."""
    clean_hwc = clean[:3].transpose(1, 2, 0)
    recon_hwc = recon[:3].transpose(1, 2, 0)
    full_psnr = peak_signal_noise_ratio(clean_hwc, recon_hwc, data_range=1.0)
    full_ssim = structural_similarity(clean_hwc, recon_hwc, data_range=1.0, channel_axis=-1)

    m = mask[..., None]
    if m.sum() > 0:
        mse_hole = ((clean_hwc - recon_hwc) ** 2 * m).sum() / (m.sum() * clean_hwc.shape[-1] + 1e-8)
        hole_psnr = 10 * np.log10(1.0 / (mse_hole + 1e-8))
    else:
        hole_psnr = float("nan")
    return full_psnr, full_ssim, hole_psnr


# ---------- UI ----------

def main():
    st.title("🛰️ TerraClear — LISS-IV Generative Cloud Removal")
    st.caption("Bharatiya Antariksh Hackathon 2026 — Challenge 2")

    mode = st.radio("Mode", ["📤 Upload your own image", "📡 Random demo patch (with ground-truth metrics)"], horizontal=True)

    with st.sidebar:
        cloud_threshold = st.slider("Cloud brightness threshold", 0.5, 1.0, 0.85, 0.05)
        st.info("Used for real uploads only — demo mode generates its own mask from the synthetic cloud.")

    if mode == "📤 Upload your own image":
        uploaded = st.file_uploader("Upload LISS-IV tile", type=["tif", "tiff", "png", "jpg", "npy"])
        if uploaded is not None:
            raw = uploaded.getvalue()
            is_npy = uploaded.name.lower().endswith(".npy")
            try:
                if is_npy:
                    arr = read_npy(raw)
                    if arr.shape[0] == 5:
                        mask, cloudy = arr[0], arr[1:5]
                    else:
                        cloudy, mask = arr, cloud_mask_from_bands(arr, cloud_threshold)
                else:
                    bands = read_bands(raw)
                    cloudy = build_model_inputs(bands)
                    mask = cloud_mask_from_bands(bands, cloud_threshold)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("☁️ Original")
                st.image(to_rgb_uint8(cloudy))
            with col2:
                st.subheader("🌍 Reconstructed")
                placeholder = st.empty()
                placeholder.info("Click Run to reconstruct.")

            if st.button("🚀 Run Reconstruction"):
                with st.spinner("Running inference..."):
                    recon = run_tiled_inference(cloudy, mask)
                    placeholder.image(to_rgb_uint8(recon))
                st.caption("No ground truth available for uploaded images — qualitative result only.")

    else:
        patches = sorted(glob.glob(str(DEMO_DIR / "*.npy")))
        if not patches:
            st.warning(
                f"No demo patches found in `{DEMO_DIR.name}/`. Add a few known-clear "
                f".npy patches (4×H×W, normalized [0,1]) from your training set to that "
                f"folder and redeploy."
            )
        else:
            if st.button("📡 Get Random Patch"):
                st.session_state["demo_patch"] = random.choice(patches)

            if "demo_patch" in st.session_state:
                with open(st.session_state["demo_patch"], "rb") as f:
                    clean = read_npy(f.read())
                clean = clean[:4]  # in case a patch has a 5th (mask) channel bundled in
                cloudy, mask = add_synthetic_cloud(clean)

                with st.spinner("Running inference..."):
                    recon = run_tiled_inference(cloudy, mask)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader("✅ Clean (ground truth)")
                    st.image(to_rgb_uint8(clean))
                with c2:
                    st.subheader("☁️ Synthetic Cloudy")
                    st.image(to_rgb_uint8(cloudy))
                with c3:
                    st.subheader("🌍 Reconstructed")
                    st.image(to_rgb_uint8(recon))

                full_psnr, full_ssim, hole_psnr = compute_metrics(clean, recon, mask)
                m1, m2, m3 = st.columns(3)
                m1.metric("Full-image PSNR", f"{full_psnr:.2f} dB")
                m2.metric("Full-image SSIM", f"{full_ssim:.3f}")
                m3.metric("PSNR inside cloud region", f"{hole_psnr:.2f} dB")
                st.caption(
                    "Cloud-region PSNR is the metric that matters — it only scores the pixels "
                    "the model actually had to invent, not the visible pixels it just passed through."
                )


if __name__ == "__main__":
    main()