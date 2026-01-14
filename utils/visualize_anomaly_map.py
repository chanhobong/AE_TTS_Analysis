#!/usr/bin/env python3
"""
visualize_anomaly_map.py
- Generate anomaly heatmap from Flow Matching model
- Visualize anomaly map overlaid on CT images
- Key visualization for the paper!

Functions:
1. generate_anomaly_heatmap: Compute anomaly score map in image space
2. overlay_heatmap_on_ct: Overlay heatmap on CT slices for visualization
"""

import os
import sys
import argparse
from typing import Tuple, Optional

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from data.dataset import CTROIVolumeDataset, load_nii

# Import models
from models import LatentVelocityEstimator
from models.flow_matching import compute_anomaly_score_mse_multi_t

# Import AE utilities
import importlib.util
spec = importlib.util.spec_from_file_location("probe_linear_classifier", 
    os.path.join(os.path.dirname(__file__), "probe_linear_classifier.py"))
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)

Tiny3DAE = probe_module.Tiny3DAE
load_model = probe_module.load_model
build_ds = probe_module.build_ds


@torch.no_grad()
def compute_anomaly_score_map_latent(
    flow_matching_model: nn.Module,
    latent_z: torch.Tensor,
    n_samples: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute anomaly score map in latent space.
    
    For each spatial location in latent space, compute the anomaly score
    by averaging MSE errors across multiple (t, z_1) samples.
    
    Args:
        flow_matching_model: Trained LatentVelocityEstimator
        latent_z: (1, C, D, H, W) latent tensor
        n_samples: Number of (t, z_1) samples to average over
        device: Device
    
    Returns:
        anomaly_score_map: (1, C, D, H, W) anomaly score map in latent space
    """
    if device is None:
        device = next(flow_matching_model.parameters()).device
    
    flow_matching_model.eval()
    latent_z = latent_z.to(device)
    
    batch_size = latent_z.shape[0]
    
    # Sample multiple (t, z_1) pairs
    t_samples = torch.rand(n_samples, batch_size, device=device)  # (n_samples, B)
    z_1_samples = torch.randn(n_samples, batch_size, *latent_z.shape[1:], device=device)  # (n_samples, B, C, D, H, W)
    
    # Compute MSE errors for each sample (keep spatial dimensions)
    error_maps = []
    for i in range(n_samples):
        t_i = t_samples[i]  # (B,)
        z_1_i = z_1_samples[i]  # (B, C, D, H, W)
        
        # Interpolate: z_t = (1-t)z + t*z_1
        t_i_expanded = t_i[:, None, None, None, None]  # (B, 1, 1, 1, 1)
        z_t_i = (1 - t_i_expanded) * latent_z + t_i_expanded * z_1_i
        
        # Predict velocity
        v_pred_i = flow_matching_model(z_t_i, t_i)  # (B, C, D, H, W)
        
        # Target velocity: u_t = z_1 - z
        u_t_i = z_1_i - latent_z  # (B, C, D, H, W)
        
        # MSE error per spatial location (keep all dimensions)
        error_i = F.mse_loss(v_pred_i, u_t_i, reduction='none')  # (B, C, D, H, W)
        error_maps.append(error_i)
    
    # Stack: (n_samples, B, C, D, H, W)
    error_maps = torch.stack(error_maps, dim=0)
    
    # Average over n_samples: (B, C, D, H, W)
    anomaly_score_map = error_maps.mean(dim=0)
    
    return anomaly_score_map


@torch.no_grad()
def generate_anomaly_heatmap(
    original_ct_image: np.ndarray,
    latent_z: torch.Tensor,
    flow_matching_model: nn.Module,
    ae_decoder: nn.Module,
    device: torch.device,
    n_samples: int = 10,
) -> np.ndarray:
    """
    Generate anomaly heatmap in image space.
    
    Args:
        original_ct_image: Original 3D CT image (numpy array, (Z, Y, X) = (64, 128, 128))
        latent_z: Latent variable z (Tensor, (1, C, D, H, W) = (1, 128, 8, 16, 16))
        flow_matching_model: Trained LatentVelocityEstimator
        ae_decoder: Trained Tiny3DAE decoder module
        device: PyTorch device
        n_samples: Number of (t, z_1) samples for score computation
    
    Returns:
        anomaly_heatmap_3d: Anomaly heatmap in image space (numpy array, (Z, Y, X) = (64, 128, 128))
    """
    flow_matching_model.eval()
    ae_decoder.eval()
    
    # Step 1: Compute anomaly score map in latent space
    # Shape: (1, C, D, H, W) = (1, 128, 8, 16, 16)
    anomaly_score_map_latent = compute_anomaly_score_map_latent(
        flow_matching_model, latent_z, n_samples=n_samples, device=device
    )
    
    # Step 2: Decode to image space using AE decoder
    # Input: (1, 128, 8, 16, 16) -> Output: (1, 1, 64, 128, 128)
    anomaly_heatmap_image_raw = ae_decoder(anomaly_score_map_latent)  # (1, 1, Z, Y, X)
    
    # Step 3: Convert to numpy and squeeze
    anomaly_heatmap_image_raw = anomaly_heatmap_image_raw.squeeze(0).squeeze(0).cpu().numpy()  # (Z, Y, X)
    
    # Step 4: Min-Max normalization to [0, 1]
    min_val = anomaly_heatmap_image_raw.min()
    max_val = anomaly_heatmap_image_raw.max()
    if max_val > min_val:
        anomaly_heatmap_3d = (anomaly_heatmap_image_raw - min_val) / (max_val - min_val)
    else:
        anomaly_heatmap_3d = np.zeros_like(anomaly_heatmap_image_raw)
    
    return anomaly_heatmap_3d


def overlay_heatmap_on_ct(
    original_ct_image: np.ndarray,
    anomaly_heatmap_3d: np.ndarray,
    slice_idx: int,
    alpha: float = 0.5,
    cmap: str = "jet",
    figsize: Tuple[int, int] = (12, 6),
    mask_3d: Optional[np.ndarray] = None,
    mask_color: str = "cyan",
    mask_linewidth: float = 1.5,
) -> plt.Figure:
    """
    Overlay anomaly heatmap on CT image slice with optional segmentation mask.
    
    Args:
        original_ct_image: Original 3D CT image (numpy array, (Z, Y, X))
        anomaly_heatmap_3d: Anomaly heatmap (numpy array, (Z, Y, X))
        slice_idx: Slice index to visualize (int)
        alpha: Transparency for heatmap overlay (float, 0~1)
        cmap: Colormap for heatmap (str, default: "jet")
        figsize: Figure size (tuple)
        mask_3d: Optional segmentation mask (numpy array, (Z, Y, X))
        mask_color: Color for mask contour (str, default: "cyan")
        mask_linewidth: Line width for mask contour (float, default: 1.5)
    
    Returns:
        fig: Matplotlib Figure object
    """
    # Get slice
    ct_slice = original_ct_image[slice_idx]  # (Y, X)
    heatmap_slice = anomaly_heatmap_3d[slice_idx]  # (Y, X)
    
    # Get mask slice if available
    mask_slice = None
    if mask_3d is not None:
        mask_slice = mask_3d[slice_idx]  # (Y, X)
        mask_slice = (mask_slice > 0).astype(float)  # Binarize
    
    # Create figure with side-by-side and overlay views
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Helper function to draw mask contour
    def draw_mask_contour(ax, mask_slice, color, linewidth):
        if mask_slice is not None and mask_slice.sum() > 0:
            # Find contours
            contours = plt.contour(mask_slice, levels=[0.5], colors=color, linewidths=linewidth)
            return contours
        return None
    
    # 1. Original CT
    ax1 = axes[0]
    im1 = ax1.imshow(ct_slice, cmap="gray")
    draw_mask_contour(ax1, mask_slice, mask_color, mask_linewidth)
    ax1.set_title(f"Original CT (Slice {slice_idx})", fontsize=12, fontweight='bold')
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 2. Anomaly Heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(heatmap_slice, cmap=cmap, vmin=0, vmax=1)
    draw_mask_contour(ax2, mask_slice, mask_color, mask_linewidth)
    ax2.set_title(f"Anomaly Heatmap (Slice {slice_idx})", fontsize=12, fontweight='bold')
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, label="Anomaly Score")
    
    # 3. Overlay
    ax3 = axes[2]
    im3 = ax3.imshow(ct_slice, cmap="gray")
    im3_overlay = ax3.imshow(heatmap_slice, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    draw_mask_contour(ax3, mask_slice, mask_color, mask_linewidth)
    ax3.set_title(f"Overlay (α={alpha})", fontsize=12, fontweight='bold')
    ax3.axis("off")
    plt.colorbar(im3_overlay, ax=ax3, fraction=0.046, label="Anomaly Score")
    
    plt.tight_layout()
    
    return fig


def visualize_multiple_slices(
    original_ct_image: np.ndarray,
    anomaly_heatmap_3d: np.ndarray,
    slice_indices: list,
    alpha: float = 0.5,
    cmap: str = "jet",
    out_path: str = None,
    mask_3d: Optional[np.ndarray] = None,
    mask_color: str = "cyan",
    mask_linewidth: float = 1.5,
) -> plt.Figure:
    """
    Visualize multiple slices in a grid.
    
    Args:
        original_ct_image: Original 3D CT image (numpy array, (Z, Y, X))
        anomaly_heatmap_3d: Anomaly heatmap (numpy array, (Z, Y, X))
        slice_indices: List of slice indices to visualize
        alpha: Transparency for heatmap overlay
        cmap: Colormap for heatmap
        out_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib Figure object
    """
    n_slices = len(slice_indices)
    n_cols = min(4, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Helper function to draw mask contour
    def draw_mask_contour(ax, mask_slice, color, linewidth):
        if mask_slice is not None and mask_slice.sum() > 0:
            contours = plt.contour(mask_slice, levels=[0.5], colors=color, linewidths=linewidth)
            return contours
        return None
    
    for idx, slice_idx in enumerate(slice_indices):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        
        ct_slice = original_ct_image[slice_idx]
        heatmap_slice = anomaly_heatmap_3d[slice_idx]
        
        # Get mask slice if available
        mask_slice = None
        if mask_3d is not None:
            mask_slice = mask_3d[slice_idx]  # (Y, X)
            mask_slice = (mask_slice > 0).astype(float)  # Binarize
        
        # CT only
        ax1 = axes[row, col]
        ax1.imshow(ct_slice, cmap="gray")
        draw_mask_contour(ax1, mask_slice, mask_color, mask_linewidth)
        ax1.set_title(f"CT Slice {slice_idx}", fontsize=10)
        ax1.axis("off")
        
        # Overlay
        ax2 = axes[row, col + 1]
        ax2.imshow(ct_slice, cmap="gray")
        ax2.imshow(heatmap_slice, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
        draw_mask_contour(ax2, mask_slice, mask_color, mask_linewidth)
        ax2.set_title(f"Overlay Slice {slice_idx}", fontsize=10)
        ax2.axis("off")
    
    # Hide unused subplots
    for idx in range(n_slices, n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")
    
    plt.suptitle("Anomaly Heatmap Visualization (Multiple Slices)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {out_path}")
    
    return fig


def build_argparser():
    p = argparse.ArgumentParser()
    
    # Model paths
    p.add_argument("--ae_model_path", type=str, required=True, help="AE model path (for encoder and decoder)")
    p.add_argument("--fm_model_path", type=str, required=True, help="Flow Matching model path")
    
    # Data
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True, help="Test CSV for selecting samples")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    
    # Visualization
    p.add_argument("--n_samples", type=int, default=10, help="Number of (t, z_1) samples for score computation")
    p.add_argument("--sample_indices", type=int, nargs="+", default=None,
                   help="Specific sample indices to visualize (None = visualize first 5)")
    p.add_argument("--slice_indices", type=int, nargs="+", default=None,
                   help="Specific slice indices (None = middle slices)")
    p.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay transparency")
    p.add_argument("--cmap", type=str, default="jet", help="Colormap for heatmap")
    
    # Mask visualization
    p.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz",
                   help="Mask filename to overlay (e.g., heart_ventricle_left_roi.nii.gz)")
    p.add_argument("--mask_color", type=str, default="cyan", help="Color for mask contour")
    p.add_argument("--mask_linewidth", type=float, default=1.5, help="Line width for mask contour")
    
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs/anomaly_maps")
    
    return p


def main():
    args = build_argparser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("ANOMALY MAP VISUALIZATION")
    print("=" * 60)
    print(f"AE Model: {args.ae_model_path}")
    print(f"FM Model: {args.fm_model_path}")
    print(f"n_samples: {args.n_samples}")
    print("=" * 60)
    
    # Load models
    print("\n--- Loading Models ---")
    ae_model = load_model(args.ae_model_path, device)
    ae_encoder = ae_model.enc
    ae_decoder = ae_model.dec
    
    checkpoint_fm = torch.load(args.fm_model_path, map_location=device)
    fm_model = LatentVelocityEstimator(
        in_channels=128,
        base_channels=64,
        time_dim=128,
        num_res_blocks=4,
    ).to(device)
    fm_model.load_state_dict(checkpoint_fm["model"])
    
    print("✅ Models loaded")
    
    # Build test dataset
    print("\n--- Building Test Dataset ---")
    ds_test_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz,
                          include_csv=args.test_csv, name="TTS(test)")
    ds_test_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                           include_csv=args.test_csv, name="Normal(test)")
    
    # Select samples to visualize
    if args.sample_indices is None:
        # Visualize first 5 samples (mix of TTS and Normal)
        sample_indices = list(range(min(5, len(ds_test_tts) + len(ds_test_norm))))
        datasets_to_use = [ds_test_tts if i < len(ds_test_tts) else ds_test_norm for i in sample_indices]
        dataset_indices = [i if i < len(ds_test_tts) else i - len(ds_test_tts) for i in sample_indices]
    else:
        sample_indices = args.sample_indices
        datasets_to_use = [ds_test_tts if i < len(ds_test_tts) else ds_test_norm for i in sample_indices]
        dataset_indices = [i if i < len(ds_test_tts) else i - len(ds_test_tts) for i in sample_indices]
    
    # Slice indices
    Z, Y, X = expected_shape_xyz[::-1]  # (Z, Y, X) = (64, 128, 128)
    if args.slice_indices is None:
        slice_indices = [Z // 4, Z // 2, 3 * Z // 4]  # Quarter, middle, three-quarter
    else:
        slice_indices = args.slice_indices
    
    print(f"✅ Will visualize {len(sample_indices)} samples")
    print(f"   Slice indices: {slice_indices}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each sample
    print("\n--- Generating Anomaly Maps ---")
    for sample_idx, (ds, ds_idx) in enumerate(zip(datasets_to_use, dataset_indices)):
        sample = ds[ds_idx]
        patient_id = sample.get("patient_id", f"sample_{sample_idx}")
        label = sample.get("y", -1)
        
        print(f"\n[{sample_idx+1}/{len(sample_indices)}] Processing: {patient_id} (label={label})")
        
        # Get CT image and convert to numpy
        x = sample["x"]  # (1, Z, Y, X)
        ct_image = x.squeeze(0).numpy()  # (Z, Y, X)
        
        # Load mask if available
        mask_3d = None
        ct_path = sample.get("ct_path", None)
        if ct_path and args.mask_filename and args.mask_filename.strip():
            # Construct mask path: same directory as CT file
            ct_dir = os.path.dirname(ct_path)
            mask_path = os.path.join(ct_dir, args.mask_filename)
            if os.path.exists(mask_path):
                try:
                    mask_xyz = load_nii(mask_path)  # (X, Y, Z) from NIfTI
                    # Convert to (Z, Y, X) to match CT image
                    mask_3d = np.transpose(mask_xyz, (2, 1, 0))  # (Z, Y, X)
                    # Ensure same shape as CT
                    if mask_3d.shape != ct_image.shape:
                        print(f"  ⚠️  Mask shape {mask_3d.shape} != CT shape {ct_image.shape}, skipping mask")
                        mask_3d = None
                    else:
                        print(f"  ✅ Loaded mask: {args.mask_filename}")
                except Exception as e:
                    print(f"  ⚠️  Failed to load mask: {e}")
            else:
                print(f"  ℹ️  Mask file not found: {mask_path}")
        
        # Extract latent z
        x_tensor = x.unsqueeze(0).to(device)  # (1, 1, Z, Y, X)
        with torch.no_grad():
            latent_z = ae_encoder(x_tensor)  # (1, C, D, H, W)
        
        # Generate anomaly heatmap
        anomaly_heatmap = generate_anomaly_heatmap(
            ct_image, latent_z, fm_model, ae_decoder, device, n_samples=args.n_samples
        )
        
        print(f"  Anomaly heatmap range: [{anomaly_heatmap.min():.4f}, {anomaly_heatmap.max():.4f}]")
        
        # Visualize multiple slices
        fig = visualize_multiple_slices(
            ct_image, anomaly_heatmap, slice_indices,
            alpha=args.alpha, cmap=args.cmap,
            mask_3d=mask_3d, mask_color=args.mask_color, mask_linewidth=args.mask_linewidth
        )
        
        out_path = os.path.join(args.out_dir, f"anomaly_map_{patient_id}_label{label}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Saved: {out_path}")
        
        # Also save individual slice overlays
        for slice_idx in slice_indices:
            fig_single = overlay_heatmap_on_ct(
                ct_image, anomaly_heatmap, slice_idx,
                alpha=args.alpha, cmap=args.cmap,
                mask_3d=mask_3d, mask_color=args.mask_color, mask_linewidth=args.mask_linewidth
            )
            out_path_single = os.path.join(
                args.out_dir, f"anomaly_map_{patient_id}_label{label}_slice{slice_idx:03d}.png"
            )
            fig_single.savefig(out_path_single, dpi=300, bbox_inches='tight')
            plt.close(fig_single)
    
    print(f"\n✅ All visualizations saved to: {args.out_dir}")
    print("✅ Visualization completed!")


if __name__ == "__main__":
    main()
