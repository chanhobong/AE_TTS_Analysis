#!/usr/bin/env python3
"""
train_ae_with_split.py
- Train a 3D autoencoder using train.csv split
- Uses include_patient_csv to filter dataset to train split only
- Save 3-view PNGs (axial/coronal/sagittal) for input vs recon

Example:
  python train_ae_with_split.py \
    --tts_root /path/to/TTS_RM_V2 \
    --normal_root /path/to/normal_RM_V2 \
    --metadata_csv /path/to/Combined_Labels.csv \
    --train_csv /path/to/train.csv \
    --epochs 50 --batch_size 4
"""

import os
import sys
import random
import argparse
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
# This allows importing from data.dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # AE_TTS directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn

# Check PyTorch installation
try:
    import torch.utils.data
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Error importing torch: {e}")
    raise

# Import data utilities
try:
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
except ImportError as e:
    print(f"Error importing from torch.utils.data: {e}")
    print("Trying alternative import method...")
    # Alternative: import individually
    import torch.utils.data as data_module
    Dataset = data_module.Dataset
    DataLoader = data_module.DataLoader
    ConcatDataset = data_module.ConcatDataset

import matplotlib.pyplot as plt

from data.dataset import CTROIVolumeDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def save_all_axial_grid(model, dataset: Dataset, out_path: str, device: torch.device, max_samples: int = None):
    """
    Save all training samples' axial views in a single grid image.
    Each sample shows input and reconstruction side by side.
    
    Args:
        model: Trained model
        dataset: Full training dataset
        out_path: Output file path
        device: Device to run inference on
        max_samples: Maximum number of samples to include (None = all)
    """
    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)
    
    # Calculate grid size: we want 2 columns (input, recon) and rows for all samples
    n_cols = 2  # Input | Recon
    n_rows = n_total
    
    # Calculate figure size (smaller images)
    fig_width = 12
    fig_height = max(8, n_rows * 0.3)  # Adjust height based on number of samples
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_total == 1:
        axes = axes.reshape(1, -1)
    
    print(f"📊 Generating grid with {n_total} samples...")
    
    for idx in range(n_total):
        s = dataset[idx]
        pid = s.get("patient_id", f"idx{idx}")
        y = int(s["y"])
        x = s["x"]  # (1, Z, Y, X)
        x_b = x.unsqueeze(0).to(device)  # (1, 1, Z, Y, X)
        
        x_hat = model(x_b).cpu().squeeze(0).squeeze(0).numpy()  # (Z, Y, X)
        x_in = x.squeeze(0).numpy()  # (Z, Y, X)
        
        # Get axial view
        Z, Y, X = x_in.shape
        z_mid = Z // 2
        axial_in = x_in[z_mid]  # (Y, X)
        axial_rec = x_hat[z_mid]  # (Y, X)
        
        # Plot input
        ax_in = axes[idx, 0]
        ax_in.imshow(axial_in, cmap="gray")
        ax_in.set_title(f"Input: {pid[:15]} (case={y})", fontsize=8)
        ax_in.axis("off")
        
        # Plot reconstruction
        ax_rec = axes[idx, 1]
        ax_rec.imshow(axial_rec, cmap="gray")
        ax_rec.set_title(f"Recon: {pid[:15]}", fontsize=8)
        ax_rec.axis("off")
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{n_total} samples...")
    
    plt.suptitle(f"All Training Samples - Axial View (Total: {n_total})", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved grid image: {out_path}")


def build_ds(root_dir, metadata_csv, expected_xyz, include_csv="", name=""):
    """Build dataset with optional include_patient_csv filter."""
    kwargs = dict(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        expected_shape_xyz=tuple(expected_xyz),
        return_patient_id=True,
    )
    if include_csv:
        kwargs["include_patient_csv"] = include_csv

    ds = CTROIVolumeDataset(**kwargs)
    if name:
        print(f"📦 {name} size: {len(ds)}")
    return ds


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--tts_root", type=str, required=True, help="Root directory for TTS dataset")
    p.add_argument("--normal_root", type=str, required=True, help="Root directory for Normal dataset")
    p.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file")
    p.add_argument("--train_csv", type=str, required=True, help="Path to train.csv split file")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64], metavar=("X", "Y", "Z"))

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="./outputs/train_ae")
    p.add_argument("--save_recons_every", type=int, default=10, help="Save recon pngs every N epochs (also saves at end)")
    p.add_argument("--preview_n", type=int, default=5, help="Number of samples to save recon previews")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")

    expected_shape_xyz = tuple(args.expected_shape)

    # Check train_csv exists
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"train_csv not found: {args.train_csv}")

    print("=" * 60)
    print("TRAINING WITH TRAIN.CSV SPLIT")
    print("=" * 60)
    print(f"tts_root     : {args.tts_root}")
    print(f"normal_root  : {args.normal_root}")
    print(f"metadata_csv : {args.metadata_csv}")
    print(f"train_csv    : {args.train_csv}")
    print(f"expected_xyz : {expected_shape_xyz}")
    print(f"epochs       : {args.epochs}")
    print(f"batch_size   : {args.batch_size}")
    print("=" * 60)

    # Build datasets using train.csv filter
    ds_tts = build_ds(
        args.tts_root,
        args.metadata_csv,
        expected_shape_xyz,
        include_csv=args.train_csv,
        name="TTS(train)"
    )
    ds_norm = build_ds(
        args.normal_root,
        args.metadata_csv,
        expected_shape_xyz,
        include_csv=args.train_csv,
        name="Normal(train)"
    )

    if len(ds_tts) == 0 or len(ds_norm) == 0:
        raise RuntimeError("Either TTS or Normal dataset is empty. Check train.csv and folder structure.")

    # Combine datasets
    ds_combined = ConcatDataset([ds_tts, ds_norm])
    print(f"✅ Combined train dataset size: {len(ds_combined)} (TTS={len(ds_tts)}, Normal={len(ds_norm)})")

    # DataLoader
    loader = DataLoader(
        ds_combined,
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
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optim, device)
        print(f"[epoch {epoch:03d}/{args.epochs}] mse={loss:.6f}")

        if args.save_recons_every > 0 and (epoch % args.save_recons_every == 0):
            recon_dir = os.path.join(args.out_dir, f"recons_epoch{epoch:03d}")
            print(f"--- saving recon pngs to {recon_dir}")
            # Save preview_n samples (individual 3-view images)
            preview_indices = list(range(min(args.preview_n, len(ds_combined))))
            recon_and_save(model, ds_combined, preview_indices, recon_dir, device)
            
            # Save all training samples' axial views in one grid image
            grid_path = os.path.join(args.out_dir, f"all_axial_grid_epoch{epoch:03d}.png")
            print(f"--- saving all axial grid to {grid_path}")
            save_all_axial_grid(model, ds_combined, grid_path, device)

    # Save final recon previews
    recon_dir = os.path.join(args.out_dir, "recons_final")
    print(f"\n--- saving final recon pngs to {recon_dir}")
    preview_indices = list(range(min(args.preview_n, len(ds_combined))))
    recon_and_save(model, ds_combined, preview_indices, recon_dir, device)
    
    # Save final all axial grid
    final_grid_path = os.path.join(args.out_dir, "all_axial_grid_final.png")
    print(f"--- saving final all axial grid to {final_grid_path}")
    save_all_axial_grid(model, ds_combined, final_grid_path, device)

    # Save weights
    ckpt_path = os.path.join(args.out_dir, "model.pth")
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "epoch": args.epochs,
    }, ckpt_path)
    print(f"\n💾 saved checkpoint: {ckpt_path}")
    print("✅ Training completed!")


if __name__ == "__main__":
    main()
