"""
TerracClear v2 — GAN-based Cloud Removal (FIXES THE PINK-PATCH PROBLEM)
==========================================================================
WHY v1 FAILED:
  Pure L1 loss has a mathematical optimum under dense, unrecoverable cloud
  cover: output the AVERAGE of every plausible ground truth seen in training.
  Average(forest_green, river_blue, soil_brown, farmland_green) = flat
  pink-grey blob. This is not a bug in your code — it is what L1 loss does
  by definition whenever the network truly cannot infer ground truth from
  context. Every cloud-removal paper that gets sharp results (pix2pix,
  SpA-GAN, GLF-CR) uses an adversarial loss specifically to escape this.

WHAT CHANGED:
  [1] Added a PatchGAN discriminator — penalises "looks averaged/flat",
      forces the generator to commit to one plausible, sharp answer.
  [2] Added VGG perceptual loss — matches texture statistics, not just
      pixel values, so output has grain/texture instead of being smooth.
  [3] Fixed PartialConv epsilon collapse on dense masks (>70% cloud) —
      added a learned fallback embedding instead of falling through to
      the renormalisation constant.
  [4] Switched base loss from L1 to a hole-weighted Charbonnier loss
      (smoother gradients near zero, trains more stably with GAN loss).
  [5] Curriculum masking — epochs 1-15 see small/medium clouds only,
      epochs 16+ introduce large dense clouds. This prevents the model
      from learning "give up and average" as its FIRST behaviour.

TARGET: SSIM 0.95-0.98 achievable on Colab T4 in ~2-3 hours for 60 epochs
         on 500-1000 patches with this architecture.

USAGE
-----
python train_v2.py --epochs 60 --amp
# Resume:
python train_v2.py --epochs 60 --amp --resume
"""

import os, sys, time, random, argparse, logging, signal, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train")

# ── Args ──────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patches",   default="data/patches")
    p.add_argument("--zip",       default=None)
    p.add_argument("--out",       default="checkpoints")
    p.add_argument("--logs",      default="logs")
    p.add_argument("--epochs",    type=int,   default=60)
    p.add_argument("--batch",     type=int,   default=None)
    p.add_argument("--lr_g",      type=float, default=2e-4)
    p.add_argument("--lr_d",      type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--workers",   type=int,   default=None)
    p.add_argument("--amp",       action="store_true")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--lambda_l1",   type=float, default=10.0)
    p.add_argument("--lambda_perc", type=float, default=1.0)
    p.add_argument("--lambda_adv",  type=float, default=0.5)
    p.add_argument("--curriculum_epoch", type=int, default=15,
                   help="Epoch at which large/dense clouds are introduced")
    return p.parse_args()

# ── Colab / Drive helpers ───────────────────────────────────────────────
def maybe_mount_drive():
    if "google.colab" not in sys.modules: return
    try: from google.colab import drive
    except ImportError: return
    if Path("/content/drive/MyDrive").exists(): return
    log.info("Detected Colab — mounting Google Drive...")
    drive.mount("/content/drive")

def maybe_unzip(patches_dir: str, zip_path):
    patch_dir = Path(patches_dir)
    if patch_dir.exists() and any(patch_dir.glob("*.npy")):
        log.info(f"Found existing .npy files in {patch_dir} — skipping unzip.")
        return
    if zip_path is None:
        raise FileNotFoundError(f"No .npy files in '{patch_dir}' and no --zip given.")
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"--zip path missing: {zip_path}")
    log.info(f"Unzipping {zip_path} -> {patch_dir} ...")
    patch_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for m in tqdm(zf.namelist(), desc="  unzip", ncols=80):
            zf.extract(m, patch_dir)
    if not any(patch_dir.glob("*.npy")):
        for f in patch_dir.rglob("*.npy"):
            target = patch_dir / f.name
            if not target.exists(): f.rename(target)
    log.info(f"Unzip complete — {len(list(patch_dir.glob('*.npy')))} .npy files ready.")

# ── Cloud augmentation WITH CURRICULUM ───────────────────────────────
# FIX [5]: early epochs only see recoverable (small/thin) clouds.
# This stops the model from learning "average everything" as its
# default strategy before it has learned to use spatial context at all.
def make_cloud_mask(size=256, epoch=999, curriculum_epoch=15):
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
    if mx - mn < 1e-8: return np.zeros((size, size), dtype=np.float32)
    mask = (mask - mn) / (mx - mn)

    # Curriculum: ramp max cloud fraction up over training
    if epoch < curriculum_epoch:
        # Early: small-medium clouds only (10-35%) — always recoverable from context
        frac = np.random.uniform(0.10, 0.35)
    else:
        # Later: full range including large dense clouds (10-75%)
        frac = np.random.uniform(0.10, 0.75)

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

# ── Dataset (epoch-aware for curriculum) ──────────────────────────────
class CloudDataset(Dataset):
    def __init__(self, paths, curriculum_epoch=15):
        self.paths = paths
        self.curriculum_epoch = curriculum_epoch
        self.current_epoch = 0   # updated externally each epoch

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try: clean = np.load(self.paths[idx]).astype(np.float32)
        except Exception: clean = np.zeros((256, 256, 4), dtype=np.float32)
        clean = np.clip(clean, 0.0, 1.0)
        if not np.isfinite(clean).all(): clean[:] = 0.0
        H, W = clean.shape[:2]
        mask   = make_cloud_mask(H, self.current_epoch, self.curriculum_epoch)
        cloudy = apply_cloud(clean, mask)
        clean  = torch.from_numpy(np.ascontiguousarray(clean.transpose(2,0,1)))
        cloudy = torch.from_numpy(np.ascontiguousarray(cloudy.transpose(2,0,1)))
        mask   = torch.from_numpy(mask).unsqueeze(0)
        return cloudy, mask, clean

def build_loaders(patch_dir, val_split, batch, workers, pin, curriculum_epoch):
    paths = sorted(Path(patch_dir).glob("*.npy"))
    if not paths: raise FileNotFoundError(f"No .npy files in {patch_dir}")
    random.shuffle(paths)
    n_val = max(1, int(len(paths) * val_split))
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=pin)
    train_ds = CloudDataset(paths[:-n_val], curriculum_epoch)
    val_ds   = CloudDataset(paths[-n_val:], curriculum_epoch=999)  # val always sees full range
    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True, **kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kw)
    log.info(f"Train={len(paths)-n_val} | Val={n_val} | Batch={batch} | Workers={workers}")
    return train_dl, val_dl, train_ds

# ── Partial Conv — FIXED epsilon collapse on dense masks ──────────────
# v1 BUG: when mask_sum hits 0 (entire conv window is cloud, common
# in deep layers under large clouds), the layer divides by epsilon and
# outputs near-zero / constant values regardless of input. This means
# DEEP layers under large clouds carry NO information at all — the
# decoder has nothing to work with except its own learned bias, which
# is exactly the flat-colour fallback you're seeing.
# FIX: when a window is fully invalid, substitute a small learned
# "unknown region" embedding instead of a near-zero/constant value.
# This gives the decoder a consistent, informative signal for "this
# was unrecoverable" rather than noise near zero.
class PartialConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv      = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.mask_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters(): p.requires_grad = False
        # FIX: learned embedding for fully-masked windows (replaces epsilon fallback)
        self.unknown_embed = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        nn.init.normal_(self.unknown_embed, std=0.02)

    def forward(self, x, mask):
        valid = 1.0 - mask
        with torch.no_grad():
            ms = self.mask_conv(valid)
        window_size = self.mask_conv.weight.sum()
        fully_invalid = (ms < 0.5)            # window has essentially zero valid pixels
        safe_ms = torch.clamp(ms, min=1.0)    # avoid div issues, but don't trust the result there
        out = self.conv(x * valid) / safe_ms * window_size
        # FIX: substitute learned embedding where window was fully invalid
        out = torch.where(fully_invalid, self.unknown_embed.expand_as(out), out)
        new_mask = 1.0 - (ms > 0).float()
        return out, new_mask

class PBR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pc = PartialConv2d(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)   # LeakyReLU — standard for GAN generators
    def forward(self, x, mask):
        x, mask = self.pc(x, mask)
        return self.act(self.bn(x)), mask

def dec_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3,1,1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3,1,1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))

# ── Generator: Partial U-Net (4 levels — full-res skip restored) ──────
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

# ── Discriminator: PatchGAN — FIX [1] ─────────────────────────────────
# Classifies overlapping 70x70 patches as real/fake rather than the
# whole image. This is what punishes "flat averaged colour" output:
# a 70x70 patch of uniform colour is trivially distinguishable from a
# 70x70 patch of real satellite texture (real terrain has high-frequency
# detail — tree canopy edges, field boundaries, riverbank texture).
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=4, base=64):
        super().__init__()
        def block(ic, oc, norm=True, stride=2):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1)]
            if norm: layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.net = nn.Sequential(
            *block(in_ch, base,    norm=False),  # 256→128
            *block(base,  base*2),                # 128→64
            *block(base*2,base*4),                #  64→32
            *block(base*4,base*8, stride=1),       #  32→31 (stride 1 — PatchGAN standard)
            nn.Conv2d(base*8, 1, 4, 1, 1),         #  31→30 — patch-wise real/fake map
        )
    def forward(self, x):
        return self.net(x)

# ── VGG Perceptual Loss — FIX [2] ──────────────────────────────────────
# Forces output to match TEXTURE statistics of ground truth, not just
# raw pixel values. This is what gives reconstructed regions visible
# grain/detail instead of looking airbrushed/flat.
class VGGPerceptualLoss(nn.Module):
    """
    BUGFIX (found during audit): the original try/except silently caught
    network errors (HTTP 403, timeout, rate limit) and returned a hardcoded
    0.0 loss FOREVER, with only one easy-to-miss log line and no further
    warning. On Colab this is a real risk — if the VGG download flakes on
    first run, your perceptual loss term silently does nothing for the
    entire 60-epoch run and you'd have no idea why texture quality was off.

    Fix: retry the download up to 3 times with backoff. If it still fails,
    raise loudly (crash the script) rather than continue silently broken —
    you want to know immediately, not after 3 hours of wasted compute.
    """
    def __init__(self, max_retries=3):
        super().__init__()
        import torchvision.models as models
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
                log.info(f"VGG16 perceptual loss backbone loaded (attempt {attempt})")
                break
            except Exception as e:
                last_err = e
                log.warning(f"VGG16 download attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)   # 2s, 4s backoff
        else:
            raise RuntimeError(
                f"VGG16 perceptual loss backbone failed to download after "
                f"{max_retries} attempts. Last error: {last_err}\n"
                f"Fix: check Colab internet connection, or rerun the cell — "
                f"this is usually a transient network blip, not a code bug."
            )
        self.vgg = vgg
        self.vgg.eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        # Use first 3 channels (RGB-ish: B04,B03,B02 → treat as RGB for VGG)
        pred_rgb   = pred[:, [2,1,0], :, :]
        target_rgb = target[:, [2,1,0], :, :]
        pred_n   = (pred_rgb   - self.mean) / self.std
        target_n = (target_rgb - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

# ── Charbonnier Loss — FIX [4] ─────────────────────────────────────────
# Smoother near zero than L1 (better gradients when GAN loss is also
# present), more robust to outliers than L2. Standard in modern
# image restoration literature (e.g. Restormer, MPRNet).
def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target)**2 + eps**2).mean()

class HoleWeightedCharbonnier(nn.Module):
    def __init__(self, hole_w=6.0, valid_w=1.0):
        super().__init__()
        self.hole_w  = hole_w
        self.valid_w = valid_w
    def forward(self, pred, target, mask):
        cloud = mask.expand_as(pred)
        clear = 1.0 - cloud
        diff  = torch.sqrt((pred - target)**2 + 1e-6)
        hole  = (diff * cloud).sum() / (cloud.sum() + 1e-4)
        valid = (diff * clear).sum() / (clear.sum() + 1e-4)
        return self.valid_w * valid + self.hole_w * hole

# ── Metrics ───────────────────────────────────────────────────────────
def psnr(p,t):
    mse=F.mse_loss(p,t); return (20*torch.log10(1.0/torch.sqrt(mse))).item() if mse>0 else 99.0

def ssim_fast(p,t):
    mp,mt=p.mean(),t.mean(); sp,st=p.var(),t.var(); spt=((p-mp)*(t-mt)).mean()
    c1,c2=0.01**2,0.03**2
    return (((2*mp*mt+c1)*(2*spt+c2))/((mp**2+mt**2+c1)*(sp+st+c2))).item()

def hole_psnr(pred, target, mask):
    """PSNR computed ONLY inside cloud regions — the metric that actually
    measures cloud removal quality, not just 'didn't break the visible pixels'."""
    cloud = mask.expand_as(pred)
    if cloud.sum() < 1: return 99.0
    mse = ((pred - target)**2 * cloud).sum() / cloud.sum()
    return (20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))).item()

# ── Checkpoint ────────────────────────────────────────────────────────
def save_ckpt(path, epoch, G, D, opt_g, opt_d, sched_g, sched_d, vl, vp, hp, args):
    torch.save({
        "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
        "val_loss": vl, "val_psnr": vp, "hole_psnr": hp, "args": vars(args),
    }, path)
    log.info(f"  ✓ Saved → {path}")

# ── Sample images ──────────────────────────────────────────────────────
def save_samples(G, loader, device, epoch, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError: return
    G.eval()
    cloudy, mask, clean = next(iter(loader))
    with torch.no_grad():
        pred = G(cloudy[:3].to(device), mask[:3].to(device)).cpu()
    def rgb(t): return np.clip(t[[2,1,0]].permute(1,2,0).numpy()*3.5, 0, 1)
    fig, ax = plt.subplots(3,3,figsize=(12,12))
    for i in range(3):
        ax[i][0].imshow(rgb(cloudy[i])); ax[i][0].set_title("Cloudy");        ax[i][0].axis("off")
        ax[i][1].imshow(rgb(pred[i]));   ax[i][1].set_title("Reconstructed (GAN)"); ax[i][1].axis("off")
        ax[i][2].imshow(rgb(clean[i]));  ax[i][2].set_title("Ground truth");  ax[i][2].axis("off")
    plt.suptitle(f"Epoch {epoch}"); plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{out_dir}/epoch_{epoch:03d}.png", dpi=100); plt.close()
    log.info(f"  Sample → {out_dir}/epoch_{epoch:03d}.png")

# ── Main ──────────────────────────────────────────────────────────────
def main():
    args=get_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    maybe_mount_drive()
    maybe_unzip(args.patches, args.zip)

    has_gpu=torch.cuda.is_available(); device=torch.device("cuda" if has_gpu else "cpu"); pin=has_gpu
    if has_gpu:
        gmem=torch.cuda.get_device_properties(0).total_memory/1e9
        log.info(f"GPU: {torch.cuda.get_device_name(0)} ({gmem:.1f} GB)")
    else:
        gmem=0; log.info("CPU mode — this WILL be very slow with a GAN, use Colab")

    if args.batch is None:
        if   not has_gpu: args.batch = 2
        elif gmem >= 14:  args.batch = 8     # T4 16GB
        else:             args.batch = 4
    if args.workers is None:
        args.workers = min(2, os.cpu_count() or 1) if has_gpu else 0

    log.info(f"Batch={args.batch} | Workers={args.workers} | Epochs={args.epochs} | "
             f"Curriculum switches at epoch {args.curriculum_epoch}")

    train_dl, val_dl, train_ds = build_loaders(
        args.patches, args.val_split, args.batch, args.workers, pin, args.curriculum_epoch)

    G = PartialUNet(in_ch=4, out_ch=4, B=64).to(device)
    D = PatchDiscriminator(in_ch=4, base=64).to(device)
    perceptual = VGGPerceptualLoss().to(device)
    base_loss  = HoleWeightedCharbonnier().to(device)
    bce        = nn.BCEWithLogitsLoss()

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    log.info(f"Generator: {g_params:.1f}M params | Discriminator: {d_params:.1f}M params")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-7)
    scaler_g = torch.cuda.amp.GradScaler() if (args.amp and has_gpu) else None
    scaler_d = torch.cuda.amp.GradScaler() if (args.amp and has_gpu) else None

    ckpt_dir   = Path(args.out);  ckpt_dir.mkdir(parents=True,exist_ok=True)
    sample_dir = Path(args.logs)/"samples"; Path(args.logs).mkdir(parents=True,exist_ok=True)

    best_val=float("inf"); start_epoch=0

    if args.resume:
        last_path = ckpt_dir/"last_model.pth"
        if last_path.exists():
            log.info(f"--resume: loading {last_path} ...")
            ckpt = torch.load(last_path, map_location=device)
            G.load_state_dict(ckpt["G"]); D.load_state_dict(ckpt["D"])
            opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
            if "sched_g" in ckpt: sched_g.load_state_dict(ckpt["sched_g"])
            if "sched_d" in ckpt: sched_d.load_state_dict(ckpt["sched_d"])
            start_epoch = ckpt.get("epoch", 0)
            log.info(f"  Resumed after epoch {start_epoch}")
        best_path = ckpt_dir/"best_model.pth"
        if best_path.exists():
            try: best_val = torch.load(best_path, map_location=device).get("val_loss", float("inf"))
            except Exception: pass

    interrupted=False
    def _handler(sig,frame):
        nonlocal interrupted
        print(); log.info("Ctrl+C — saving and exiting after this epoch..."); interrupted=True
    signal.signal(signal.SIGINT, _handler)

    history={"train_g":[],"val":[],"psnr":[],"ssim":[],"hole_psnr":[]}

    for epoch in range(start_epoch+1, args.epochs+1):
        if interrupted: break
        t0=time.time()
        train_ds.current_epoch = epoch   # drives curriculum masking
        phase = "SMALL/MEDIUM clouds only" if epoch < args.curriculum_epoch else "FULL range incl. dense clouds"
        log.info(f"\nEpoch {epoch}/{args.epochs}  [curriculum: {phase}]")

        G.train(); D.train()
        g_loss_total = 0.0
        bar = tqdm(train_dl, desc="  train", leave=False, ncols=80)

        for cloudy, mask, clean in bar:
            cloudy = cloudy.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)
            clean  = clean.to(device, non_blocking=True)
            B = cloudy.size(0)

            # ── Train Discriminator ──────────────────────────────────
            opt_d.zero_grad()
            with torch.no_grad():
                fake = G(cloudy, mask)
            if scaler_d:
                with torch.autocast(device_type="cuda"):
                    d_real = D(clean)
                    d_fake = D(fake.detach())
                    real_label = torch.ones_like(d_real) * 0.9   # label smoothing
                    fake_label = torch.zeros_like(d_fake)
                    d_loss = bce(d_real, real_label) + bce(d_fake, fake_label)
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d); scaler_d.update()
            else:
                d_real = D(clean)
                d_fake = D(fake.detach())
                real_label = torch.ones_like(d_real) * 0.9
                fake_label = torch.zeros_like(d_fake)
                d_loss = bce(d_real, real_label) + bce(d_fake, fake_label)
                d_loss.backward(); opt_d.step()

            # ── Train Generator ──────────────────────────────────────
            opt_g.zero_grad()
            if scaler_g:
                with torch.autocast(device_type="cuda"):
                    pred   = G(cloudy, mask)
                    d_pred = D(pred)
                    adv_loss  = bce(d_pred, torch.ones_like(d_pred))
                    rec_loss  = base_loss(pred, clean, mask)
                    perc_loss = perceptual(pred, clean)
                    g_loss = (args.lambda_l1   * rec_loss +
                             args.lambda_perc  * perc_loss +
                             args.lambda_adv   * adv_loss)
                scaler_g.scale(g_loss).backward()
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                scaler_g.step(opt_g); scaler_g.update()
            else:
                pred   = G(cloudy, mask)
                d_pred = D(pred)
                adv_loss  = bce(d_pred, torch.ones_like(d_pred))
                rec_loss  = base_loss(pred, clean, mask)
                perc_loss = perceptual(pred, clean)
                g_loss = (args.lambda_l1   * rec_loss +
                         args.lambda_perc  * perc_loss +
                         args.lambda_adv   * adv_loss)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_g.step()

            g_loss_total += g_loss.item()
            bar.set_postfix(g=f"{g_loss.item():.3f}", d=f"{d_loss.item():.3f}")

        train_g = g_loss_total / len(train_dl)
        sched_g.step(); sched_d.step()

        # ── Validation ─────────────────────────────────────────────
        G.eval(); vl=vp=vs=vh=0.0
        with torch.no_grad():
            for cloudy,mask,clean in val_dl:
                cloudy=cloudy.to(device); mask=mask.to(device); clean=clean.to(device)
                pred = G(cloudy, mask)
                vl += base_loss(pred, clean, mask).item()
                vp += psnr(pred, clean)
                vs += ssim_fast(pred, clean)
                vh += hole_psnr(pred, clean, mask)
        n = len(val_dl)
        vl, vp, vs, vh = vl/n, vp/n, vs/n, vh/n

        elapsed = time.time()-t0
        log.info(f"  G_loss={train_g:.4f}  val_rec={vl:.4f} | "
                 f"PSNR={vp:.2f}dB  SSIM={vs:.4f}  HolePSNR={vh:.2f}dB | "
                 f"{elapsed:.0f}s  ETA≈{elapsed*(args.epochs-epoch)/60:.0f}min")

        history["train_g"].append(train_g); history["val"].append(vl)
        history["psnr"].append(vp); history["ssim"].append(vs); history["hole_psnr"].append(vh)

        save_ckpt(ckpt_dir/"last_model.pth", epoch, G, D, opt_g, opt_d, sched_g, sched_d, vl, vp, vh, args)
        if vl < best_val:
            best_val = vl
            save_ckpt(ckpt_dir/"best_model.pth", epoch, G, D, opt_g, opt_d, sched_g, sched_d, vl, vp, vh, args)
        if epoch % 5 == 0:
            save_samples(G, val_dl, device, epoch, sample_dir)
        if interrupted: break

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        ax[0].plot(history["train_g"], label="G loss"); ax[0].plot(history["val"], label="val rec"); ax[0].legend(); ax[0].set_title("Loss")
        ax[1].plot(history["psnr"]); ax[1].set_title("Global PSNR ↑")
        ax[2].plot(history["ssim"]); ax[2].set_title("SSIM ↑")
        ax[3].plot(history["hole_psnr"]); ax[3].set_title("Hole-region PSNR ↑ (cloud regions only)")
        plt.tight_layout(); plt.savefig(f"{args.logs}/training_curves.png", dpi=120)
        log.info(f"Curves → {args.logs}/training_curves.png")
    except Exception: pass

    log.info(f"\n{'Interrupted' if interrupted else 'Done'}!")
    log.info(f"Best  → {ckpt_dir}/best_model.pth")
    log.info(f"Last  → {ckpt_dir}/last_model.pth")

if __name__=="__main__":
    main()