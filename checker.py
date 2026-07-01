"""
TerracClear — ONNX Model Checker
===================================
Loads your exported .onnx model, runs it on RANDOM real patches from your
dataset, and shows side-by-side: cloudy input | model output | ground truth.

This is the actual visual sanity check — metrics can lie, your eyes can't.
Specifically checks for the pink-patch failure mode from before.

USAGE
-----
python checker.py --onnx model.onnx --patches data/patches --n 6
python checker.py --onnx model.onnx --patches data/patches --n 6 --dense_only
    # ^ forces only LARGE dense clouds — the hardest case, the one that broke v1
"""

import argparse, random, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("checker")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",     default="model.onnx")
    p.add_argument("--patches",  default="data/patches")
    p.add_argument("--n",        type=int, default=6, help="number of random patches to test")
    p.add_argument("--out",      default="checker_results.png")
    p.add_argument("--seed",     type=int, default=None, help="omit for a different random set each run")
    p.add_argument("--dense_only", action="store_true",
                   help="force large dense clouds (40-75%%) — the hard failure case")
    return p.parse_args()


def make_cloud_mask(size=256, frac_range=(0.15, 0.65)):
    """Same generator as training — kept self-contained so this file has no
    import dependency on train_v2.py and works even if that file changes."""
    import cv2
    mask = np.zeros((size, size), dtype=np.float32)
    scale = np.random.uniform(30, 100)
    amp, freq = 1.0, 1.0
    for _ in range(6):
        layer = np.random.randn(size, size).astype(np.float32)
        k = max(3, int(scale / freq))
        k = k if k % 2 == 1 else k + 1
        layer = cv2.GaussianBlur(layer, (k, k), 0)
        mask += amp * layer
        amp *= 0.5; freq *= 2.0
    mn, mx = mask.min(), mask.max()
    if mx - mn < 1e-8:
        return np.zeros((size, size), dtype=np.float32)
    mask = (mask - mn) / (mx - mn)
    frac = np.random.uniform(*frac_range)
    thresh = float(np.percentile(mask, (1 - frac) * 100))
    binary = (mask >= thresh).astype(np.float32)
    return cv2.GaussianBlur(binary, (15, 15), 0)


def apply_cloud(clean, mask):
    H, W, C = clean.shape
    b = np.random.uniform(0.72, 0.96)
    cloud = np.full((H, W, C), b, dtype=np.float32)
    cloud += np.random.randn(H, W, C).astype(np.float32) * 0.03
    m3 = mask[:, :, None]
    return np.clip(cloud * m3 + clean * (1 - m3), 0.0, 1.0)


def to_rgb(arr_chw):
    """[C,H,W] float32 -> displayable RGB. Uses B04,B03,B02 (indices 2,1,0) as R,G,B."""
    rgb = arr_chw[[2,1,0]].transpose(1, 2, 0)
    return np.clip(rgb * 3.5, 0, 1)


def hole_psnr(pred, target, mask):
    """PSNR computed ONLY inside the cloud mask — the metric that actually
    tells you if reconstruction worked, not just 'didn't break visible pixels'."""
    mask3 = np.repeat(mask[None, :, :], pred.shape[0], axis=0)
    if mask3.sum() < 1:
        return float("nan")
    mse = ((pred - target) ** 2 * mask3).sum() / mask3.sum()
    if mse <= 0:
        return 99.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def global_ssim(pred, target):
    mp, mt = pred.mean(), target.mean()
    sp, st = pred.var(), target.var()
    spt = ((pred - mp) * (target - mt)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2*mp*mt + c1) * (2*spt + c2)
    den = (mp**2 + mt**2 + c1) * (sp + st + c2)
    return num / den


def main():
    args = get_args()

    # ── Load ONNX runtime ────────────────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime not installed. Run: pip install onnxruntime\n"
            "(or onnxruntime-gpu if you want GPU inference)"
        )

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            f"Run export_onnx.py first to create it from your trained checkpoint."
        )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" \
                else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    log.info(f"Loaded {onnx_path} | providers: {sess.get_providers()}")

    input_names = [i.name for i in sess.get_inputs()]
    log.info(f"Model inputs: {input_names}")
    expected = {"cloudy_image", "cloud_mask"}
    if not expected.issubset(set(input_names)):
        raise ValueError(
            f"ONNX model has unexpected input names: {input_names}\n"
            f"Expected: {expected}\n"
            f"This usually means the .onnx was exported from a different script version."
        )

    # ── Find patches ─────────────────────────────────────────────────────
    patch_dir = Path(args.patches)
    paths = sorted(patch_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy patches found in {patch_dir}")
    log.info(f"Found {len(paths)} patches in {patch_dir}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    n = min(args.n, len(paths))
    chosen = random.sample(paths, n)

    frac_range = (0.40, 0.75) if args.dense_only else (0.15, 0.65)
    if args.dense_only:
        log.info("--dense_only set: testing LARGE clouds (40-75%) — the hard failure case")

    # ── Run inference on each chosen patch ──────────────────────────────
    results = []
    for path in chosen:
        try:
            clean = np.load(path).astype(np.float32)   # [H, W, C]
        except Exception as e:
            log.warning(f"Skipping corrupt file {path.name}: {e}")
            continue

        if clean.ndim != 3 or clean.shape[2] < 4:
            log.warning(f"Skipping {path.name}: unexpected shape {clean.shape}")
            continue

        clean = np.clip(clean, 0.0, 1.0)
        H, W = clean.shape[:2]

        mask = make_cloud_mask(H, frac_range)
        cloudy = apply_cloud(clean, mask)

        # [H,W,C] -> [1,C,H,W] for ONNX
        cloudy_chw = cloudy.transpose(2, 0, 1)[None, ...].astype(np.float32)
        mask_chw   = mask[None, None, ...].astype(np.float32)
        clean_chw  = clean.transpose(2, 0, 1)

        try:
            onnx_out = sess.run(None, {
                "cloudy_image": cloudy_chw,
                "cloud_mask":   mask_chw,
            })
            pred = onnx_out[0][0]   # [C,H,W]
        except Exception as e:
            log.error(f"Inference failed on {path.name}: {e}")
            continue

        if np.isnan(pred).any() or np.isinf(pred).any():
            log.warning(f"{path.name}: model output contains NaN/Inf!")

        hp = hole_psnr(pred, clean_chw, mask)
        ss = global_ssim(pred, clean_chw)

        # ── Pink-patch detector: flag low-variance output in cloud region ──
        cloud_region_pred = pred[:, mask > 0.5]
        flat_warning = ""
        if cloud_region_pred.size > 0:
            region_std = cloud_region_pred.std()
            if region_std < 0.02:
                flat_warning = f"  ⚠ LOW VARIANCE ({region_std:.4f}) in cloud region — possible flat/averaged output"

        results.append({
            "name": path.name, "clean": clean_chw, "cloudy": cloudy_chw[0],
            "pred": pred, "mask": mask, "hole_psnr": hp, "ssim": ss,
            "cloud_frac": float(mask.mean()), "flat_warning": flat_warning,
        })

        log.info(f"{path.name}  |  cloud={mask.mean()*100:.0f}%  |  "
                f"hole_psnr={hp:.2f}dB  global_ssim={ss:.4f}{flat_warning}")

    if not results:
        log.error("No patches were successfully processed.")
        return

    # ── Summary stats ────────────────────────────────────────────────────
    avg_hole_psnr = np.nanmean([r["hole_psnr"] for r in results])
    avg_ssim      = np.mean([r["ssim"] for r in results])
    n_flat        = sum(1 for r in results if r["flat_warning"])

    log.info("")
    log.info("=" * 60)
    log.info(f"SUMMARY over {len(results)} random patches:")
    log.info(f"  Avg hole-region PSNR : {avg_hole_psnr:.2f} dB  "
             f"({'good' if avg_hole_psnr > 25 else 'needs more training' if avg_hole_psnr > 18 else 'POOR — check training'})")
    log.info(f"  Avg global SSIM      : {avg_ssim:.4f}")
    log.info(f"  Flat/pink-patch flags: {n_flat}/{len(results)}  "
             f"({'GOOD — no flat patches detected' if n_flat == 0 else 'investigate flagged patches below'})")
    log.info("=" * 60)

    # ── Visual grid ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping visual output. "
                   "Install with: pip install matplotlib")
        return

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1:
        axes = axes[None, :]

    for i, r in enumerate(results):
        axes[i][0].imshow(to_rgb(r["cloudy"]))
        axes[i][0].set_title(f"Cloudy ({r['cloud_frac']*100:.0f}% cover)")
        axes[i][0].axis("off")

        axes[i][1].imshow(to_rgb(r["pred"]))
        title = f"Model output | hole_psnr={r['hole_psnr']:.1f}dB"
        color = "red" if r["flat_warning"] else "black"
        axes[i][1].set_title(title, color=color)
        axes[i][1].axis("off")

        axes[i][2].imshow(to_rgb(r["clean"]))
        axes[i][2].set_title("Ground truth")
        axes[i][2].axis("off")

        fig.text(0.01, 1 - (i + 0.5) / n, r["name"][:24], fontsize=8, rotation=90,
                 va="center", ha="left")

    plt.tight_layout()
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    log.info(f"\nVisual grid saved → {out_path}")
    log.info("Open this PNG and look specifically at the 'Model output' column —")
    log.info("if cloud regions show texture/detail similar to ground truth, it's working.")
    log.info("If they're flat/uniform colored blobs, the model needs more epochs or the GAN loss weight needs tuning.")


if __name__ == "__main__":
    main()