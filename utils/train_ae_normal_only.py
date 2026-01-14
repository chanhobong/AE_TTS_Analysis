#!/usr/bin/env python3
"""
train_ae_normal_only.py
- Train AE using ONLY Normal data (classic anomaly detection baseline)
- This is for comparison with Mixed (TTS+Normal) training

The hypothesis: "Normal만으로 학습하면 TTS가 reconstruction error로 더 잘 튀어나오지 않을까?"

Example:
  python train_ae_normal_only.py \
    --normal_root /path/to/normal_RM_V2 \
    --metadata_csv /path/to/Combined_Labels.csv \
    --train_csv /path/to/train.csv \
    --epochs 100 --batch_size 4
"""

import os
import sys
import random
import argparse
from typing import List, Dict, Any, Tuple

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from data.dataset import CTROIVolumeDataset

# Import Tiny3DAE from train_ae_with_split
from train_ae_with_split import Tiny3DAE, set_seed, train_one_epoch, save_all_axial_grid, build_ds


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--normal_root", type=str, required=True, help="Root directory for Normal dataset")
    p.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file")
    p.add_argument("--train_csv", type=str, required=True, help="Path to train.csv split file")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64], metavar=("X", "Y", "Z"))

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="./outputs/train_ae_normal_only")
    p.add_argument("--save_recons_every", type=int, default=10, help="Save recon pngs every N epochs")
    p.add_argument("--preview_n", type=int, default=5, help="Number of samples to save recon previews")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")

    expected_shape_xyz = tuple(args.expected_shape)

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"train_csv not found: {args.train_csv}")

    print("=" * 60)
    print("TRAINING AE WITH NORMAL-ONLY DATA")
    print("=" * 60)
    print(f"normal_root  : {args.normal_root}")
    print(f"metadata_csv : {args.metadata_csv}")
    print(f"train_csv    : {args.train_csv}")
    print(f"expected_xyz : {expected_shape_xyz}")
    print(f"epochs       : {args.epochs}")
    print(f"batch_size   : {args.batch_size}")
    print("=" * 60)
    print("⚠️  NOTE: Training on NORMAL data ONLY (anomaly detection baseline)")
    print("=" * 60)

    # Build dataset using ONLY Normal data with train.csv filter
    ds_norm = build_ds(
        args.normal_root,
        args.metadata_csv,
        expected_shape_xyz,
        include_csv=args.train_csv,
        name="Normal(train, normal-only)"
    )

    if len(ds_norm) == 0:
        raise RuntimeError("Normal dataset is empty. Check train.csv and folder structure.")

    print(f"✅ Normal-only train dataset size: {len(ds_norm)}")

    # DataLoader
    loader = DataLoader(
        ds_norm,
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
            preview_indices = list(range(min(args.preview_n, len(ds_norm))))
            from train_ae_with_split import recon_and_save
            recon_and_save(model, ds_norm, preview_indices, recon_dir, device)
            
            # Save all training samples' axial views in one grid image
            grid_path = os.path.join(args.out_dir, f"all_axial_grid_epoch{epoch:03d}.png")
            print(f"--- saving all axial grid to {grid_path}")
            save_all_axial_grid(model, ds_norm, grid_path, device)

    # Save final recon previews
    recon_dir = os.path.join(args.out_dir, "recons_final")
    print(f"\n--- saving final recon pngs to {recon_dir}")
    preview_indices = list(range(min(args.preview_n, len(ds_norm))))
    from train_ae_with_split import recon_and_save
    recon_and_save(model, ds_norm, preview_indices, recon_dir, device)
    
    # Save final all axial grid
    final_grid_path = os.path.join(args.out_dir, "all_axial_grid_final.png")
    print(f"--- saving final all axial grid to {final_grid_path}")
    save_all_axial_grid(model, ds_norm, final_grid_path, device)

    # Save weights
    ckpt_path = os.path.join(args.out_dir, "model.pth")
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "epoch": args.epochs,
        "training_type": "normal_only",  # Mark this as normal-only training
    }, ckpt_path)
    print(f"\n💾 saved checkpoint: {ckpt_path}")
    print("✅ Training completed!")
    print("\n📝 Next step: Compare with Mixed model using compare_normal_vs_mixed.py")


if __name__ == "__main__":
    main()
