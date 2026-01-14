#!/usr/bin/env python3
"""
train_ddpm.py
- Train DDPM model on Normal latent space
- Extract spatial latents from trained AE encoder
- Train LatentVelocityEstimator as noise predictor using DDPM
"""

import os
import sys
import argparse
import random
from typing import Tuple

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.dataset import CTROIVolumeDataset
from models import LatentVelocityEstimator
from models.ddpm import NoiseSchedule, compute_ddpm_loss
from latent_extraction import LatentDataset

# Import AE model
import importlib.util
spec = importlib.util.spec_from_file_location("probe_linear_classifier", 
    os.path.join(os.path.dirname(__file__), "probe_linear_classifier.py"))
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)

Tiny3DAE = probe_module.Tiny3DAE
load_model = probe_module.load_model
build_ds = probe_module.build_ds


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                   noise_schedule: NoiseSchedule, device: torch.device) -> float:
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for batch in loader:
        z_0 = batch["z"].to(device)  # (B, C, D, H, W)
        
        # Compute DDPM loss
        loss, _ = compute_ddpm_loss(model, z_0, noise_schedule, device)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * z_0.shape[0]
        n_samples += z_0.shape[0]
    
    return total_loss / max(n_samples, 1)


def build_argparser():
    p = argparse.ArgumentParser(description="Train DDPM model on Normal latent space")
    
    # Data paths
    p.add_argument("--ae_model_path", type=str, required=True,
                   help="Path to trained AE model (Normal-only)")
    p.add_argument("--normal_root", type=str, required=True,
                   help="Root directory for Normal dataset")
    p.add_argument("--metadata_csv", type=str, required=True,
                   help="Path to metadata CSV file")
    p.add_argument("--train_csv", type=str, required=True,
                   help="Path to train.csv split file")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                   metavar=("X", "Y", "Z"), help="Expected volume shape (X, Y, Z)")
    
    # Model config
    p.add_argument("--base_channels", type=int, default=64,
                   help="Base channels for LatentVelocityEstimator (64~128 recommended)")
    p.add_argument("--time_dim", type=int, default=128,
                   help="Time embedding dimension")
    p.add_argument("--num_res_blocks", type=int, default=4,
                   help="Number of residual blocks")
    
    # DDPM config
    p.add_argument("--timesteps", type=int, default=1000,
                   help="Number of diffusion timesteps")
    p.add_argument("--schedule_type", type=str, default="cosine",
                   choices=["linear", "cosine"],
                   help="Noise schedule type (linear or cosine)")
    
    # Training config
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size (small for M5 MacBook)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--num_workers", type=int, default=2,
                   help="Number of data loader workers")
    
    # Other
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--out_dir", type=str, default="./outputs/ddpm",
                   help="Output directory")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs")
    
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    
    # Device: MPS for M5 MacBook
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"✅ Device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("DDPM TRAINING")
    print("=" * 60)
    print(f"AE Model: {args.ae_model_path}")
    print(f"Normal Root: {args.normal_root}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Base Channels: {args.base_channels}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Schedule Type: {args.schedule_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)
    
    # Load AE model
    print("\n--- Loading AE Model ---")
    ae_model = load_model(args.ae_model_path, device)
    
    # Check encoder output shape
    with torch.no_grad():
        test_input = torch.randn(1, 1, *expected_shape_xyz[::-1]).to(device)  # (1, 1, Z, Y, X)
        test_z = ae_model.enc(test_input)
        print(f"Encoder output shape: {test_z.shape}")
    
    latent_channels = test_z.shape[1]
    print(f"✅ Latent channels: {latent_channels}")
    
    # Build Normal dataset (train split only)
    print("\n--- Building Normal Dataset (Train Split) ---")
    ds_normal = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                        include_csv=args.train_csv, name="Normal(train)")
    
    if len(ds_normal) == 0:
        raise RuntimeError("Normal dataset is empty. Check train.csv and folder structure.")
    
    # Create latent dataset
    print("\n--- Creating Latent Dataset ---")
    latent_ds = LatentDataset(ae_model, ds_normal, device)
    
    # DataLoader
    loader = DataLoader(
        latent_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Create noise schedule (on device)
    print("\n--- Creating Noise Schedule ---")
    noise_schedule = NoiseSchedule(timesteps=args.timesteps, schedule_type=args.schedule_type, device=device)
    print(f"✅ Noise schedule created ({args.timesteps} timesteps, {args.schedule_type})")
    
    # Create DDPM model (reuse LatentVelocityEstimator as noise predictor)
    print("\n--- Creating DDPM Model ---")
    model = LatentVelocityEstimator(
        in_channels=latent_channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        num_res_blocks=args.num_res_blocks,
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, noise_schedule, device)
        print(f"[epoch {epoch:03d}/{args.epochs}] loss={loss:.6f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "noise_schedule": {
                    "timesteps": args.timesteps,
                    "schedule_type": args.schedule_type,
                },
                "args": vars(args),
            }, ckpt_path)
            print(f"  💾 Saved: {ckpt_path}")
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "noise_schedule": {
                    "timesteps": args.timesteps,
                    "schedule_type": args.schedule_type,
                },
                "args": vars(args),
            }, best_path)
    
    # Save final model
    final_path = os.path.join(args.out_dir, "model.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": args.epochs,
        "loss": loss,
        "noise_schedule": {
            "timesteps": args.timesteps,
            "schedule_type": args.schedule_type,
        },
        "args": vars(args),
    }, final_path)
    print(f"\n💾 Saved final model: {final_path}")
    print("✅ Training completed!")


if __name__ == "__main__":
    main()
