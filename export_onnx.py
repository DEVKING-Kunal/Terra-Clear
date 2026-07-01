"""
TerracClear — Export trained Generator to ONNX
=================================================
Run this AFTER training completes, to produce a portable model file
you can load anywhere (no PyTorch/CUDA needed at inference time).

USAGE
-----
python export_onnx.py --checkpoint checkpoints/best_model.pth --out model.onnx
"""

import argparse, logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("export")

# ── Model definition — MUST exactly match train_v2.py's generator ─────
class PartialConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv      = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.mask_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters(): p.requires_grad = False
        self.unknown_embed = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x, mask):
        valid = 1.0 - mask
        with torch.no_grad():
            ms = self.mask_conv(valid)
        window_size = self.mask_conv.weight.sum()
        fully_invalid = (ms < 0.5)
        safe_ms = torch.clamp(ms, min=1.0)
        out = self.conv(x * valid) / safe_ms * window_size
        out = torch.where(fully_invalid, self.unknown_embed.expand_as(out), out)
        new_mask = 1.0 - (ms > 0).float()
        return out, new_mask

class PBR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pc = PartialConv2d(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x, mask):
        x, mask = self.pc(x, mask)
        return self.act(self.bn(x)), mask

def dec_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3,1,1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3,1,1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))

class PartialUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, B=64):
        super().__init__()
        self.enc1 = PBR(in_ch, B)
        self.enc2 = PBR(B,     B*2)
        self.enc3 = PBR(B*2,   B*4)
        self.enc4 = PBR(B*4,   B*8)
        self.bottle = nn.Sequential(
            nn.Conv2d(B*8, B*8, 3,1,1), nn.BatchNorm2d(B*8), nn.LeakyReLU(0.2, inplace=True))
        self.up3  = nn.ConvTranspose2d(B*8, B*4, 2, 2)
        self.dec3 = dec_block(B*4 + B*8, B*4)
        self.up2  = nn.ConvTranspose2d(B*4, B*2, 2, 2)
        self.dec2 = dec_block(B*2 + B*4, B*2)
        self.up1  = nn.ConvTranspose2d(B*2, B,   2, 2)
        self.dec1 = dec_block(B   + B*2, B)
        self.head = nn.Conv2d(B, out_ch, 1)

    def forward(self, x, mask):
        e1,m1 = self.enc1(x,  mask)
        e2,m2 = self.enc2(F.max_pool2d(e1,2), F.max_pool2d(m1,2))
        e3,m3 = self.enc3(F.max_pool2d(e2,2), F.max_pool2d(m2,2))
        e4,m4 = self.enc4(F.max_pool2d(e3,2), F.max_pool2d(m3,2))
        b  = self.bottle(F.max_pool2d(e4,2))
        d3 = self.dec3(torch.cat([self.up3(b),  e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
        out = torch.sigmoid(self.head(d1))
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--out", default="model.onnx")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = get_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first with train_v2.py, or check the path."
        )

    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # train_v2.py saves generator weights under key "G"
    if "G" in ckpt:
        state_dict = ckpt["G"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]   # fallback for v1-style checkpoints
        log.warning("Checkpoint uses v1 format ('model' key) — exporting v1-style weights. "
                   "These won't include the GAN improvements from v2.")
    else:
        raise KeyError(f"Checkpoint has no 'G' or 'model' key. Found keys: {list(ckpt.keys())}")

    model = PartialUNet(in_ch=4, out_ch=4, B=64)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            raise RuntimeError(
                "Checkpoint architecture does not match this script's PartialUNet(B=64).\n"
                "This usually means the checkpoint was trained with a different --base_channels "
                "value, or train_v2.py's model definition changed after training.\n"
                "Fix: either retrain with the default B=64, or edit the 'B=64' on the "
                "PartialUNet(...) line in this file to match what you actually trained with.\n"
                f"\nOriginal PyTorch error (truncated): {str(e)[:300]}"
            ) from e
        raise
    model.eval()

    epoch = ckpt.get("epoch", "?")
    psnr  = ckpt.get("val_psnr", "?")
    hole_psnr = ckpt.get("hole_psnr", "?")
    log.info(f"Checkpoint info — epoch: {epoch}, val_psnr: {psnr}, hole_psnr: {hole_psnr}")

    # ── Dummy inputs for tracing ────────────────────────────────────────
    dummy_x    = torch.rand(1, 4, args.patch_size, args.patch_size)
    dummy_mask = torch.zeros(1, 1, args.patch_size, args.patch_size)
    dummy_mask[:, :, 50:150, 50:150] = 1.0   # non-trivial mask so tracing covers the partial-conv branch

    log.info(f"Exporting to ONNX (opset {args.opset}, legacy exporter)...")
    out_path = Path(args.out)

    try:
        # IMPORTANT: dynamo=False forces PyTorch's legacy ONNX exporter.
        # Found during testing: PyTorch 2.x's newer dynamo-based exporter has
        # two separate bugs that break this exact export —
        #   (1) opset 17 can't downgrade the Resize op used by our bilinear
        #       interpolate fallback, silently forcing opset 18 instead
        #   (2) the external-data writer hits an "OSError: Bad file descriptor"
        #       in some environments when saving multi-file ONNX output
        # The legacy exporter (dynamo=False) does not have either issue and
        # is still fully supported — this is the same exporter most production
        # ONNX pipelines use today, not a workaround or a downgrade in quality.
        torch.onnx.export(
            model,
            (dummy_x, dummy_mask),
            str(out_path),
            input_names=["cloudy_image", "cloud_mask"],
            output_names=["reconstructed_image"],
            dynamic_axes={
                "cloudy_image":        {0: "batch", 2: "height", 3: "width"},
                "cloud_mask":          {0: "batch", 2: "height", 3: "width"},
                "reconstructed_image": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as e:
        log.error(f"ONNX export failed: {e}")
        log.error("If this is an opset-related error, try --opset 18 instead of 17.")
        log.error("If this mentions 'dynamo' or 'onnxscript', your torch version may not "
                  "support dynamo=False — try `pip install --upgrade torch` first.")
        raise

    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"Exported → {out_path} ({size_mb:.1f} MB)")

    # ── Verify the export is actually loadable and numerically correct ──
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        log.info("ONNX model structure check: PASS")

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {
            "cloudy_image": dummy_x.numpy(),
            "cloud_mask":   dummy_mask.numpy(),
        })[0]

        with torch.no_grad():
            torch_out = model(dummy_x, dummy_mask).numpy()

        import numpy as np
        max_diff = np.abs(onnx_out - torch_out).max()
        log.info(f"Max difference between PyTorch and ONNX output: {max_diff:.6f}")
        if max_diff < 1e-3:
            log.info("Numerical equivalence check: PASS — ONNX output matches PyTorch")
        else:
            log.warning(f"Numerical difference is larger than expected ({max_diff:.6f}). "
                       f"Model may still work but double-check results with checker.py.")

    except ImportError:
        log.warning("onnx/onnxruntime not installed — skipping verification. "
                   "Install with: pip install onnx onnxruntime")
        log.warning("The .onnx file was created but NOT verified to load correctly.")

    log.info("\nDone. Use checker.py to visually inspect results on real patches.")


if __name__ == "__main__":
    main()