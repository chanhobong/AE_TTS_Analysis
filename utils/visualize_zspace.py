#!/usr/bin/env python3
"""
visualize_zspace.py
- Extract z (latent space) from test set
- Visualize with PCA and UMAP
- Color by label (TTS vs Normal)
- Check if clusters are visible and separable

This is a critical sanity check for the paper!
"""

import os
import sys
import random
import argparse
from typing import Tuple

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ UMAP not installed. Install with: pip install umap-learn")

from data.dataset import CTROIVolumeDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Tiny3DAE(nn.Module):
    """Same architecture as training script."""
    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base * 2, base * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base * 4, base * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(base * 8, base * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base * 4, base * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base * 2, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat

    def encode(self, x):
        """Extract z (latent space) from encoder."""
        z = self.enc(x)
        # Global Average Pooling
        z_pooled = nn.functional.adaptive_avg_pool3d(z, (1, 1, 1))
        z_flat = z_pooled.view(z_pooled.size(0), -1)
        return z_flat


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load trained AE model."""
    print(f"📦 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if "args" in checkpoint:
        args = checkpoint["args"]
        base = args.get("base", 16) if isinstance(args, dict) else 16
    else:
        base = 16
    
    model = Tiny3DAE(in_ch=1, base=base).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"✅ Model loaded (base={base})")
    return model


@torch.no_grad()
def extract_z_and_labels(model: nn.Module, dataset, device: torch.device, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Extract z (latent) and labels from dataset."""
    model.eval()
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    
    all_z = []
    all_labels = []
    all_patient_ids = []
    
    print(f"  Extracting z from {len(dataset)} samples...")
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        pids = batch["patient_id"]
        
        z = model.encode(x)
        z_np = z.cpu().numpy()
        
        all_z.append(z_np)
        all_labels.append(y)
        all_patient_ids.extend(pids)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Processed {batch_idx + 1} batches...")
    
    z_array = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"  ✅ z shape: {z_array.shape}, labels shape: {labels.shape}")
    return z_array, labels, all_patient_ids


def build_ds(root_dir, metadata_csv, expected_xyz, include_csv="", name=""):
    """Build dataset."""
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


def plot_pca(z: np.ndarray, labels: np.ndarray, out_path: str, title_suffix: str = ""):
    """Visualize z-space with PCA."""
    print("\n--- PCA Visualization ---")
    
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(z)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
    print(f"  Total explained: {explained_var.sum():.3f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Separate by label
    tts_mask = labels == 1
    normal_mask = labels == 0
    
    ax.scatter(z_pca[normal_mask, 0], z_pca[normal_mask, 1], 
              c='blue', label='Normal (0)', alpha=0.6, s=50)
    ax.scatter(z_pca[tts_mask, 0], z_pca[tts_mask, 1], 
              c='red', label='TTS (1)', alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'Latent Space Visualization - PCA{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


def plot_umap(z: np.ndarray, labels: np.ndarray, out_path: str, title_suffix: str = ""):
    """Visualize z-space with UMAP."""
    if not HAS_UMAP:
        print("  ⚠️ Skipping UMAP (not installed)")
        return
    
    print("\n--- UMAP Visualization ---")
    
    # UMAP to 2D
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    z_umap = reducer.fit_transform(z)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Separate by label
    tts_mask = labels == 1
    normal_mask = labels == 0
    
    ax.scatter(z_umap[normal_mask, 0], z_umap[normal_mask, 1], 
              c='blue', label='Normal (0)', alpha=0.6, s=50)
    ax.scatter(z_umap[tts_mask, 0], z_umap[tts_mask, 1], 
              c='red', label='TTS (1)', alpha=0.6, s=50)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'Latent Space Visualization - UMAP{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to trained model.pth")
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True, help="Test set CSV")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64], metavar=("X", "Y", "Z"))
    
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs/zspace_vis")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("Z-SPACE VISUALIZATION (PCA/UMAP)")
    print("=" * 60)
    print(f"model_path : {args.model_path}")
    print(f"test_csv   : {args.test_csv}")
    print("=" * 60)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Build test dataset
    print("\n--- Building Test Dataset ---")
    ds_test_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz, 
                          include_csv=args.test_csv, name="TTS(test)")
    ds_test_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz, 
                           include_csv=args.test_csv, name="Normal(test)")
    ds_test = ConcatDataset([ds_test_tts, ds_test_norm])
    
    print(f"\n✅ Test size: {len(ds_test)}")
    print(f"   - TTS: {len(ds_test_tts)}")
    print(f"   - Normal: {len(ds_test_norm)}")
    
    # Extract z and labels
    print("\n--- Extracting Latent Space (z) ---")
    z, labels, patient_ids = extract_z_and_labels(model, ds_test, device, args.batch_size)
    
    # Label distribution
    n_tts = (labels == 1).sum()
    n_normal = (labels == 0).sum()
    print(f"\nLabel distribution:")
    print(f"  TTS (1): {n_tts}")
    print(f"  Normal (0): {n_normal}")
    
    # Visualize
    os.makedirs(args.out_dir, exist_ok=True)
    
    # PCA
    pca_path = os.path.join(args.out_dir, "zspace_pca_test.png")
    plot_pca(z, labels, pca_path, " (Test Set)")
    
    # UMAP
    if HAS_UMAP:
        umap_path = os.path.join(args.out_dir, "zspace_umap_test.png")
        plot_umap(z, labels, umap_path, " (Test Set)")
    
    # Save z and labels for further analysis
    np.save(os.path.join(args.out_dir, "z_test.npy"), z)
    np.save(os.path.join(args.out_dir, "labels_test.npy"), labels)
    
    print(f"\n💾 Saved z and labels to: {args.out_dir}")
    print("\n✅ Visualization completed!")
    print("\n📊 Questions to check:")
    print("   - Are clusters visible?")
    print("   - Do TTS and Normal separate along one axis?")
    print("   - This could be Figure 1 in your paper!")


if __name__ == "__main__":
    main()
