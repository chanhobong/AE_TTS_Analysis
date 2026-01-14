#!/usr/bin/env python3
"""
sanity_ae_train.py
- Train a tiny 3D autoencoder on a small subset (e.g., 10 samples)
- Save 3-view PNGs (axial/coronal/sagittal) for input vs recon

Assumes you have CTROIVolumeDataset in LDAE_TTS/data/dataset.py and it returns:
  sample = {"x": (1, Z, Y, X) torch float, "y": int, "meta": {...}, "patient_id": str, "ct_path": str}

Example:
  python sanity_ae_train.py \
    --tts_root /Volumes/Chanho_PhD_Project/DatasetRaw/TTS_RM_V2 \
    --normal_root /Volumes/Chanho_PhD_Project/DatasetRaw/normal_RM_V2 \
    --metadata_csv /Volumes/Chanho_PhD_Project/DatasetRaw/Combined_Labels.csv \
    --n_total 10 --epochs 20 --batch_size 2
"""

import os
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt

# IMPORTANT: adjust import if your path differs
# If this script is placed next to check.py (LDAE_TTS/data/check.py),
# you may want: from data.dataset import CTROIVolumeDataset
# Here, we assume you're running inside LDAE_TTS/ and dataset.py is in data/
from data.dataset import CTROIVolumeDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CombinedDataset(Dataset):
    """A + B dataset without copying samples."""
    def __init__(self, ds_a: Dataset, ds_b: Dataset):
        self.ds_a = ds_a
        self.ds_b = ds_b

    def __len__(self):
        return len(self.ds_a) + len(self.ds_b)

    def __getitem__(self, idx):
        if idx < len(self.ds_a):
            return self.ds_a[idx]
        return self.ds_b[idx - len(self.ds_a)]


class Tiny3DAE(nn.Module):
    """
    Very small 3D AE for sanity:
    input:  (B, 1, Z, Y, X) = (B, 1, 64, 128, 128)
    output: same
    """
    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(base, base * 2, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),

            nn.Conv3d(base * 2, base * 4, kernel_size=4, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),

            nn.Conv3d(base * 4, base * 8, kernel_size=4, stride=2, padding=1),  # /8
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(base * 8, base * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(base * 4, base * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(base * 2, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(base, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # good if your input x is in [-1, 1]
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat


def pick_balanced_indices(ds_tts: Dataset, ds_norm: Dataset, n_total: int, seed: int) -> Tuple[List[int], List[int]]:
    """
    Choose about half from each class (if available).
    Returns indices for ds_tts and ds_norm separately.
    """
    rng = random.Random(seed)
    n_each = n_total // 2
    n_tts = min(len(ds_tts), n_each)
    n_norm = min(len(ds_norm), n_total - n_tts)

    idx_tts = list(range(len(ds_tts)))
    idx_norm = list(range(len(ds_norm)))

    rng.shuffle(idx_tts)
    rng.shuffle(idx_norm)

    return idx_tts[:n_tts], idx_norm[:n_norm]


def get_three_views(vol_zyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    vol_zyx: (Z, Y, X)
    returns:
      axial    = vol_zyx[z_mid, :, :]        -> (Y, X)
      coronal  = vol_zyx[:, y_mid, :]        -> (Z, X)
      sagittal = vol_zyx[:, :, x_mid]        -> (Z, Y)
    """
    Z, Y, X = vol_zyx.shape
    z_mid = Z // 2
    y_mid = Y // 2
    x_mid = X // 2

    axial = vol_zyx[z_mid]
    coronal = vol_zyx[:, y_mid, :]
    sagittal = vol_zyx[:, :, x_mid]
    return axial, coronal, sagittal


def save_three_view_png(
    out_path: str,
    x_in: np.ndarray,
    x_rec: np.ndarray,
    title: str = "",
):
    """
    x_in, x_rec: (Z, Y, X) in numpy
    Saves a 2x3 grid: [input axial, coronal, sagittal] / [recon axial, coronal, sagittal]
    """
    ax_in, co_in, sa_in = get_three_views(x_in)
    ax_re, co_re, sa_re = get_three_views(x_rec)

    fig = plt.figure(figsize=(12, 8))
    if title:
        fig.suptitle(title)

    imgs = [
        ("Input Axial", ax_in),
        ("Input Coronal", co_in),
        ("Input Sagittal", sa_in),
        ("Recon Axial", ax_re),
        ("Recon Coronal", co_re),
        ("Recon Sagittal", sa_re),
    ]

    for i, (name, img) in enumerate(imgs, start=1):
        a = fig.add_subplot(2, 3, i)
        a.imshow(img, cmap="gray")  # grayscale medical view
        a.set_title(name)
        a.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_one_epoch(model, loader, optim, device) -> float:
    model.train()
    total = 0.0
    n = 0
    loss_fn = nn.MSELoss()

    for batch in loader:
        x = batch["x"].to(device)  # (B, 1, Z, Y, X)
        optim.zero_grad(set_to_none=True)
        x_hat = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()
        optim.step()

        total += float(loss.item()) * x.shape[0]
        n += x.shape[0]

    return total / max(n, 1)


@torch.no_grad()
def recon_and_save(model, dataset: Dataset, indices: List[int], out_dir: str, device: torch.device):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    for k, idx in enumerate(indices):
        s = dataset[idx]
        pid = s.get("patient_id", f"idx{idx}")
        y = int(s["y"])
        x = s["x"]  # (1, Z, Y, X)
        x_b = x.unsqueeze(0).to(device)  # (1, 1, Z, Y, X)

        x_hat = model(x_b).cpu().squeeze(0).squeeze(0).numpy()  # (Z, Y, X)
        x_in = x.squeeze(0).numpy()  # (Z, Y, X)

        # File name
        out_path = os.path.join(out_dir, f"{k:02d}_{pid}_case{y}_3view.png")

        # Add quick stats in title (optional)
        title = f"{pid} | case={y} | in[min={x_in.min():.3f}, max={x_in.max():.3f}] | rec[min={x_hat.min():.3f}, max={x_hat.max():.3f}]"
        save_three_view_png(out_path, x_in, x_hat, title=title)

        print(f"🖼️ saved: {out_path}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--tts_root", type=str, default="/Volumes/Chanho_PhD_Project/DatasetRaw/TTS_RM_V2")
    p.add_argument("--normal_root", type=str, default="/Volumes/Chanho_PhD_Project/DatasetRaw/normal_RM_V2")
    p.add_argument("--metadata_csv", type=str, default="/Volumes/Chanho_PhD_Project/DatasetRaw/Combined_Labels.csv")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64], metavar=("X", "Y", "Z"))

    p.add_argument("--n_total", type=int, default=10, help="Total samples for sanity training (balanced if possible)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="./outputs/sanity_ae")
    p.add_argument("--save_recons_every", type=int, default=10, help="Save recon pngs every N epochs (also saves at end)")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")

    expected_shape_xyz = tuple(args.expected_shape)

    # Build per-class datasets (your folder contains only that class)
    ds_tts = CTROIVolumeDataset(
        root_dir=args.tts_root,
        metadata_csv=args.metadata_csv,
        return_patient_id=True,
        expected_shape_xyz=expected_shape_xyz,
    )
    ds_norm = CTROIVolumeDataset(
        root_dir=args.normal_root,
        metadata_csv=args.metadata_csv,
        return_patient_id=True,
        expected_shape_xyz=expected_shape_xyz,
    )

    print(f"📦 TTS size   : {len(ds_tts)}")
    print(f"📦 Normal size: {len(ds_norm)}")

    # Select a small balanced subset
    idx_tts, idx_norm = pick_balanced_indices(ds_tts, ds_norm, n_total=args.n_total, seed=args.seed)

    # Build combined subset indices in a CombinedDataset
    ds_combined = CombinedDataset(ds_tts, ds_norm)
    # Combined indices: TTS are [0..len(ds_tts)-1], Normal shift by len(ds_tts)
    comb_indices = idx_tts + [len(ds_tts) + i for i in idx_norm]

    subset = Subset(ds_combined, comb_indices)
    print(f"✅ sanity subset size: {len(subset)} (TTS={len(idx_tts)}, Normal={len(idx_norm)})")

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = Tiny3DAE(in_ch=1, base=16).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    # Train
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optim, device)
        print(f"[epoch {epoch:03d}/{args.epochs}] mse={loss:.6f}")

        if args.save_recons_every > 0 and (epoch % args.save_recons_every == 0):
            recon_dir = os.path.join(args.out_dir, f"recons_epoch{epoch:03d}")
            print(f"--- saving recon pngs to {recon_dir}")
            recon_and_save(model, subset, list(range(len(subset))), recon_dir, device)

    # Save final recon previews
    recon_dir = os.path.join(args.out_dir, f"recons_final")
    print(f"--- saving final recon pngs to {recon_dir}")
    recon_and_save(model, subset, list(range(len(subset))), recon_dir, device)

    # Save weights
    ckpt_path = os.path.join(args.out_dir, "tiny3dae.pth")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"💾 saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()