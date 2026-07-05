"""
TerraClear — LISS-IV Cloud Reconstruction
Bharatiya Antariksh Hackathon 2026 — Challenge 2
"""
import glob
import io
import os
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rasterio
import streamlit as st
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="TerraClear",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------------
st.markdown("""
<style>
    /* Base */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header strip */
    .tc-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 28px 36px 20px 36px;
        margin-bottom: 28px;
    }
    .tc-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .tc-header p {
        color: #8b949e;
        font-size: 0.88rem;
        margin: 0;
    }
    .tc-badge {
        display: inline-block;
        background: #1f6feb22;
        border: 1px solid #1f6feb55;
        color: #58a6ff;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 16px; margin-top: 20px; }
    .metric-card {
        flex: 1;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #8b949e;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 4px;
    }
    .metric-card.good .value { color: #3fb950; }
    .metric-card.warn .value { color: #d29922; }
    .metric-card.info .value { color: #58a6ff; }

    /* Image labels */
    .img-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }

    /* Section divider */
    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Note box */
    .note-box {
        background: #161b22;
        border-left: 3px solid #30363d;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 16px;
    }

    /* Mode selector override */
    div[data-testid="stRadio"] label { font-size: 0.9rem; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.onnx"
DEMO_DIR   = Path(__file__).parent / "demo_patches"
TILE       = 256
OVERLAP    = 32

# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

session = load_model()

# ---------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------
def read_bands(file_bytes: bytes) -> np.ndarray:
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        arr = src.read().astype(np.float32)
        arr = arr / max(arr.max(), 1.0)
    return arr

def read_npy(file_bytes: bytes) -> np.ndarray:
    arr = np.load(io.BytesIO(file_bytes), allow_pickle=False).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr.shape}")
    if arr.shape[0] not in (4, 5) and arr.shape[-1] in (4, 5):
        arr = arr.transpose(2, 0, 1)
    if arr.max() > 1.5:
        arr = arr / max(arr.max(), 1.0)
    return arr

def build_model_inputs(bands: np.ndarray) -> np.ndarray:
    n = bands.shape[0]
    idx = [0, 1, 2, 3] if n >= 4 else [0, 1, 2, n - 1]
    return bands[idx]

def cloud_mask_from_bands(bands: np.ndarray, threshold: float) -> np.ndarray:
    return (bands.mean(axis=0) > threshold).astype(np.float32)

# ---------------------------------------------------------------
# SYNTHETIC CLOUD GENERATOR (mirrors train_v2.py exactly)
# ---------------------------------------------------------------
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

def add_synthetic_cloud(clean: np.ndarray, seed=None):
    rng = np.random.default_rng(seed)
    c, h, w = clean.shape
    mask = make_cloud_mask(h, rng=rng)
    clean_hwc = clean.transpose(1, 2, 0)
    cloudy_hwc = apply_cloud(clean_hwc, mask, rng=rng)
    return cloudy_hwc.transpose(2, 0, 1).astype(np.float32), mask.astype(np.float32)

# ---------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------
def run_tiled_inference(cloudy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    c, h, w = cloudy.shape
    out    = np.zeros((c, h, w), dtype=np.float32)
    weight = np.zeros((h, w),    dtype=np.float32)
    step   = TILE - OVERLAP
    for y in range(0, max(h - OVERLAP, 1), step):
        for x in range(0, max(w - OVERLAP, 1), step):
            y2, x2 = min(y + TILE, h), min(x + TILE, w)
            y,  x  = max(y2 - TILE, 0), max(x2 - TILE, 0)
            tile_in   = cloudy[:, y:y+TILE, x:x+TILE][None].astype(np.float32)
            tile_mask = mask[y:y+TILE, x:x+TILE][None, None].astype(np.float32)
            result = session.run(
                ["reconstructed_image"],
                {"cloudy_image": tile_in, "cloud_mask": tile_mask}
            )[0][0]
            wy  = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE))
            w2d = np.outer(wy, wy).astype(np.float32)
            w2d /= w2d.max()
            out[:, y:y+TILE, x:x+TILE] += result * w2d
            weight[y:y+TILE, x:x+TILE] += w2d
    return out / np.clip(weight, 1e-6, None)[None]

def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr[:3].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

def compute_metrics(clean, recon, mask):
    clean_hwc = clean[:3].transpose(1, 2, 0)
    recon_hwc = recon[:3].transpose(1, 2, 0)
    full_psnr = peak_signal_noise_ratio(clean_hwc, recon_hwc, data_range=1.0)
    full_ssim = structural_similarity(clean_hwc, recon_hwc, data_range=1.0, channel_axis=-1)
    m = mask[..., None]
    if m.sum() > 0:
        mse_hole  = ((clean_hwc - recon_hwc) ** 2 * m).sum() / (m.sum() * clean_hwc.shape[-1] + 1e-8)
        hole_psnr = 10 * np.log10(1.0 / (mse_hole + 1e-8))
    else:
        hole_psnr = float("nan")
    return full_psnr, full_ssim, hole_psnr

# ---------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------
def quality_class(ssim_val):
    if ssim_val >= 0.90: return "good"
    if ssim_val >= 0.75: return "warn"
    return ""

def psnr_class(psnr_val):
    if psnr_val >= 30: return "good"
    if psnr_val >= 20: return "warn"
    return ""

def render_metrics(full_psnr, full_ssim, hole_psnr):
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card {psnr_class(full_psnr)}">
            <div class="label">Full-image PSNR</div>
            <div class="value">{full_psnr:.2f} <span style="font-size:1rem;font-weight:400">dB</span></div>
            <div class="sub">Peak signal-to-noise ratio</div>
        </div>
        <div class="metric-card {quality_class(full_ssim)}">
            <div class="label">Full-image SSIM</div>
            <div class="value">{full_ssim:.4f}</div>
            <div class="sub">Structural similarity index</div>
        </div>
        <div class="metric-card info">
            <div class="label">Cloud-region PSNR</div>
            <div class="value">{hole_psnr:.2f} <span style="font-size:1rem;font-weight:400">dB</span></div>
            <div class="sub">Restricted to occluded pixels only</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():

    # Header
    st.markdown("""
    <div class="tc-header">
        <div class="tc-badge">Bharatiya Antariksh Hackathon 2026 &mdash; Challenge 2</div>
        <h1>🛰️ TerraClear</h1>
        <p>Generative cloud removal and surface reconstruction for LISS-IV satellite imagery &mdash;
        Partial U-Net + PatchGAN, exported to ONNX, tiled inference with feather blending.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
        cloud_threshold = st.slider(
            "Cloud brightness threshold",
            min_value=0.5, max_value=1.0, value=0.85, step=0.05,
            help="Pixels brighter than this value across all bands are treated as cloud. Applies to uploaded imagery only."
        )
        st.markdown("""
        <div class="note-box">
            Threshold applies to uploaded GeoTIFF and image files only.
            Demo mode generates its own cloud mask from the training distribution.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:24px">Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="note-box">
            Architecture: Partial U-Net + PatchGAN<br>
            Loss: L1 + VGG perceptual + adversarial<br>
            Input: 256 × 256 tiles (tiled for larger scenes)<br>
            Runtime: ONNX, CPU inference
        </div>
        """, unsafe_allow_html=True)

    # Mode selector
    st.markdown('<div class="section-title">Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Select mode",
        ["Upload imagery", "Benchmarked demo"],
        label_visibility="collapsed",
        horizontal=True
    )

    st.markdown("---")

    # ---- MODE 1: Upload ----
    if mode == "Upload imagery":
        st.markdown('<div class="section-title">Input</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a LISS-IV tile or patch",
            type=["tif", "tiff", "png", "jpg", "npy"],
            help="Accepts GeoTIFF (multi-band), standard images, or .npy arrays from the training pipeline."
        )

        if uploaded is not None:
            raw    = uploaded.getvalue()
            is_npy = uploaded.name.lower().endswith(".npy")

            try:
                if is_npy:
                    arr = read_npy(raw)
                    if arr.shape[0] == 5:
                        mask, cloudy = arr[0], arr[1:5]
                    else:
                        cloudy = arr
                        mask   = cloud_mask_from_bands(arr, cloud_threshold)
                else:
                    bands  = read_bands(raw)
                    cloudy = build_model_inputs(bands)
                    mask   = cloud_mask_from_bands(bands, cloud_threshold)
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                st.stop()

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<div class="img-label">Input</div>', unsafe_allow_html=True)
                st.image(to_rgb_uint8(cloudy))
            with col2:
                st.markdown('<div class="img-label">Reconstruction</div>', unsafe_allow_html=True)
                result_slot = st.empty()
                result_slot.markdown(
                    '<div style="height:200px;background:#161b22;border:1px solid #30363d;'
                    'border-radius:8px;display:flex;align-items:center;justify-content:center;'
                    'color:#8b949e;font-size:0.85rem;">Run inference to see output</div>',
                    unsafe_allow_html=True
                )

            if st.button("Run reconstruction", type="primary"):
                with st.spinner("Running tiled inference..."):
                    recon = run_tiled_inference(cloudy, mask)
                result_slot.image(to_rgb_uint8(recon))

            st.markdown("""
            <div class="note-box">
                No ground-truth surface is available for uploaded imagery — output is qualitative only.
                Use the benchmarked demo mode for scored evaluation against a known reference.
            </div>
            """, unsafe_allow_html=True)

    # ---- MODE 2: Demo ----
    else:
        patches = sorted(glob.glob(str(DEMO_DIR / "*.npy")))

        if not patches:
            st.warning(
                f"No demo patches found in `{DEMO_DIR.name}/`. "
                "Add known-clear .npy patches (4 × H × W, normalized to [0, 1]) "
                "from the training set to that folder and redeploy."
            )
            return

        st.markdown("""
        <div class="note-box" style="margin-bottom:20px">
            Selects a random cloud-free patch from the training set, applies a synthetic cloud
            using the same distribution the model was trained on, runs reconstruction, and
            scores the result against the known ground truth. Cloud-region PSNR is the primary
            metric — it only measures the pixels the model had to invent.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Load random patch", type="primary"):
            st.session_state["demo_patch"] = random.choice(patches)

        if "demo_patch" in st.session_state:
            with open(st.session_state["demo_patch"], "rb") as f:
                clean = read_npy(f.read())
            clean        = clean[:4]
            cloudy, mask = add_synthetic_cloud(clean)

            with st.spinner("Running tiled inference..."):
                recon = run_tiled_inference(cloudy, mask)

            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                st.markdown('<div class="img-label">Ground truth</div>', unsafe_allow_html=True)
                st.image(to_rgb_uint8(clean))
            with c2:
                st.markdown('<div class="img-label">Cloud-occluded input</div>', unsafe_allow_html=True)
                st.image(to_rgb_uint8(cloudy))
            with c3:
                st.markdown('<div class="img-label">Reconstruction</div>', unsafe_allow_html=True)
                st.image(to_rgb_uint8(recon))

            st.markdown('<div class="section-title" style="margin-top:28px">Evaluation</div>', unsafe_allow_html=True)

            full_psnr, full_ssim, hole_psnr = compute_metrics(clean, recon, mask)
            render_metrics(full_psnr, full_ssim, hole_psnr)

            st.markdown("""
            <div class="note-box" style="margin-top:16px">
                Cloud-region PSNR isolates the occluded pixels — the ones the model had to reconstruct
                from context rather than copy through. Full-image metrics are inflated by the unoccluded
                regions and are less informative for this task.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
