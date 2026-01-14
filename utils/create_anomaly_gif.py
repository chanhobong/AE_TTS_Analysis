#!/usr/bin/env python3
"""
create_anomaly_gif.py
- Create GIF animations for Normal and TTS samples
- Shows: Original CT, Heatmap overlay, Mask+Heatmap overlay
- Based on visualize_anomaly_map.py
"""

import os
import sys
import argparse
from typing import Tuple, Optional
import io

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
try:
    import imageio
except ImportError:
    print("⚠️  imageio not found. Installing...")
    os.system("pip install imageio")
    import imageio

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
    """Compute anomaly score map in latent space."""
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
    """Generate anomaly heatmap in image space."""
    # Step 1: Compute anomaly score map in latent space
    anomaly_score_map_latent = compute_anomaly_score_map_latent(
        flow_matching_model, latent_z, n_samples=n_samples, device=device
    )
    
    # Step 2: Decode to image space
    anomaly_heatmap_image_raw = ae_decoder(anomaly_score_map_latent)
    anomaly_heatmap_image_raw = anomaly_heatmap_image_raw.squeeze(0).squeeze(0).cpu().numpy()  # (Z, Y, X)
    
    # Step 3: Min-Max normalization to [0, 1]
    min_val = anomaly_heatmap_image_raw.min()
    max_val = anomaly_heatmap_image_raw.max()
    if max_val > min_val:
        anomaly_heatmap_3d = (anomaly_heatmap_image_raw - min_val) / (max_val - min_val)
    else:
        anomaly_heatmap_3d = np.zeros_like(anomaly_heatmap_image_raw)
    
    return anomaly_heatmap_3d


def create_slice_image(
    ct_slice: np.ndarray,
    heatmap_slice: Optional[np.ndarray],
    mask_slice: Optional[np.ndarray],
    slice_idx: int,
    patient_id: str,
    label: int,
    show_heatmap: bool = True,
    show_mask: bool = False,
    alpha: float = 0.5,
    cmap: str = "jet",
    mask_color: str = "cyan",
    mask_linewidth: float = 1.5,
    figsize: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Create a single slice image (returns numpy array for GIF)."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show CT in grayscale
    ax.imshow(ct_slice, cmap="gray")
    
    # Overlay heatmap if requested
    if show_heatmap and heatmap_slice is not None:
        ax.imshow(heatmap_slice, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    
    # Draw mask contour if available and requested
    if show_mask and mask_slice is not None and mask_slice.sum() > 0:
        contours = plt.contour(mask_slice, levels=[0.5], colors=mask_color, linewidths=mask_linewidth)
    
    label_str = "TTS" if label == 1 else "Normal"
    title_parts = [f"{patient_id} - {label_str}", f"Slice {slice_idx:03d}"]
    if show_heatmap:
        title_parts.append("Heatmap")
    if show_mask and mask_slice is not None:
        title_parts.append("+ Mask")
    ax.set_title("\n".join(title_parts), fontsize=12, fontweight='bold')
    ax.axis("off")
    
    # Convert figure to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() for compatibility with newer matplotlib versions
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    buf = buf.reshape((height, width, 4))  # RGBA
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)
    
    return buf


def create_gif_from_slices(
    ct_image: np.ndarray,
    anomaly_heatmap: Optional[np.ndarray],
    mask_3d: Optional[np.ndarray],
    patient_id: str,
    label: int,
    out_path: str,
    show_heatmap: bool = True,
    alpha: float = 0.5,
    cmap: str = "jet",
    mask_color: str = "cyan",
    mask_linewidth: float = 1.5,
    duration: float = 0.2,
    show_mask: bool = False,
):
    """Create GIF from all slices."""
    Z, Y, X = ct_image.shape
    
    print(f"  Creating GIF with {Z} slices...")
    
    frames = []
    for slice_idx in range(Z):
        ct_slice = ct_image[slice_idx]
        
        heatmap_slice = None
        if show_heatmap and anomaly_heatmap is not None:
            heatmap_slice = anomaly_heatmap[slice_idx]
        
        mask_slice = None
        if mask_3d is not None:
            mask_slice = mask_3d[slice_idx]
            mask_slice = (mask_slice > 0).astype(float)
        
        frame = create_slice_image(
            ct_slice, heatmap_slice, mask_slice, slice_idx, patient_id, label,
            show_heatmap=show_heatmap, show_mask=show_mask,
            alpha=alpha, cmap=cmap,
            mask_color=mask_color, mask_linewidth=mask_linewidth
        )
        frames.append(frame)
        
        if (slice_idx + 1) % 10 == 0:
            print(f"    Processed {slice_idx + 1}/{Z} slices...")
    
    # Save GIF
    imageio.mimsave(out_path, frames, duration=duration, loop=0)
    print(f"  ✅ Saved GIF: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Create GIF animations for anomaly maps")
    
    # Model paths
    parser.add_argument("--ae_model_path", type=str, required=True)
    parser.add_argument("--fm_model_path", type=str, required=True)
    
    # Data paths
    parser.add_argument("--tts_root", type=str, required=True)
    parser.add_argument("--normal_root", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    
    # Sample selection
    parser.add_argument("--normal_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                       help="Indices of Normal samples to use (default: 0-4, total 5 samples)")
    parser.add_argument("--tts_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                       help="Indices of TTS samples to use (default: 0-4, total 5 samples)")
    
    # Parameters
    parser.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--n_samples", type=int, default=10, help="Number of (t, z_1) samples for score computation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay transparency")
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap for heatmap")
    
    # Mask visualization
    parser.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz",
                       help="Mask filename to overlay")
    parser.add_argument("--mask_color", type=str, default="cyan", help="Color for mask contour")
    parser.add_argument("--mask_linewidth", type=float, default=1.5, help="Line width for mask contour")
    
    # GIF parameters
    parser.add_argument("--duration", type=float, default=0.2, help="Duration per frame in seconds")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./outputs/anomaly_gifs")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("ANOMALY MAP GIF CREATION")
    print("=" * 60)
    print(f"AE Model: {args.ae_model_path}")
    print(f"FM Model: {args.fm_model_path}")
    print(f"Normal sample indices: {args.normal_indices} (total: {len(args.normal_indices)} samples)")
    print(f"TTS sample indices: {args.tts_indices} (total: {len(args.tts_indices)} samples)")
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
    
    # Build test datasets
    print("\n--- Building Test Datasets ---")
    ds_test_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz,
                          include_csv=args.test_csv, name="TTS(test)")
    ds_test_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                           include_csv=args.test_csv, name="Normal(test)")
    
    print(f"  TTS samples: {len(ds_test_tts)}")
    print(f"  Normal samples: {len(ds_test_norm)}")
    
    # Filter valid indices
    valid_normal_indices = [idx for idx in args.normal_indices if idx < len(ds_test_norm)]
    valid_tts_indices = [idx for idx in args.tts_indices if idx < len(ds_test_tts)]
    
    if len(valid_normal_indices) < len(args.normal_indices):
        invalid = set(args.normal_indices) - set(valid_normal_indices)
        print(f"⚠️  Some Normal indices are out of range: {invalid}, using valid ones: {valid_normal_indices}")
    
    if len(valid_tts_indices) < len(args.tts_indices):
        invalid = set(args.tts_indices) - set(valid_tts_indices)
        print(f"⚠️  Some TTS indices are out of range: {invalid}, using valid ones: {valid_tts_indices}")
    
    # Build samples to process list
    samples_to_process = []
    for idx in valid_normal_indices:
        samples_to_process.append((ds_test_norm, idx, "Normal"))
    for idx in valid_tts_indices:
        samples_to_process.append((ds_test_tts, idx, "TTS"))
    
    print(f"\n  Will process {len(samples_to_process)} samples:")
    print(f"    Normal: {len(valid_normal_indices)} samples (indices: {valid_normal_indices})")
    print(f"    TTS: {len(valid_tts_indices)} samples (indices: {valid_tts_indices})")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each sample
    print("\n--- Creating GIFs ---")
    for ds, idx, label_name in samples_to_process:
        sample = ds[idx]
        patient_id = sample.get("patient_id", f"sample_{idx}")
        label = sample.get("y", -1)
        
        print(f"\n[{label_name}] Processing: {patient_id} (label={label})")
        
        # Get CT image and convert to numpy
        x = sample["x"]  # (1, Z, Y, X)
        ct_image = x.squeeze(0).numpy()  # (Z, Y, X)
        
        # Load mask if available
        mask_3d = None
        ct_path = sample.get("ct_path", None)
        if ct_path and args.mask_filename and args.mask_filename.strip():
            ct_dir = os.path.dirname(ct_path)
            mask_path = os.path.join(ct_dir, args.mask_filename)
            if os.path.exists(mask_path):
                try:
                    mask_xyz = load_nii(mask_path)  # (X, Y, Z) from NIfTI
                    mask_3d = np.transpose(mask_xyz, (2, 1, 0))  # (Z, Y, X)
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
        
        # Create 3 GIFs: Original, Heatmap, Mask+Heatmap
        base_name = f"{patient_id}_label{label}"
        
        # 1. Original CT only
        print(f"\n  Creating GIF 1/3: Original CT...")
        gif_path_original = os.path.join(args.out_dir, f"{base_name}_original.gif")
        create_gif_from_slices(
            ct_image, None, None, patient_id, label,
            gif_path_original, show_heatmap=False,
            duration=args.duration, show_mask=False
        )
        
        # 2. Heatmap overlay
        print(f"\n  Creating GIF 2/3: Heatmap overlay...")
        gif_path_heatmap = os.path.join(args.out_dir, f"{base_name}_heatmap.gif")
        create_gif_from_slices(
            ct_image, anomaly_heatmap, None, patient_id, label,
            gif_path_heatmap, show_heatmap=True,
            alpha=args.alpha, cmap=args.cmap,
            duration=args.duration, show_mask=False
        )
        
        # 3. Mask + Heatmap overlay
        if mask_3d is not None:
            print(f"\n  Creating GIF 3/3: Mask + Heatmap overlay...")
            gif_path_mask = os.path.join(args.out_dir, f"{base_name}_mask_heatmap.gif")
            create_gif_from_slices(
                ct_image, anomaly_heatmap, mask_3d, patient_id, label,
                gif_path_mask, show_heatmap=True,
                alpha=args.alpha, cmap=args.cmap,
                mask_color=args.mask_color, mask_linewidth=args.mask_linewidth,
                duration=args.duration, show_mask=True
            )
        else:
            print(f"\n  ⚠️  Skipping mask GIF (mask not available)")
    
    print(f"\n✅ All GIFs saved to: {args.out_dir}")
    print("✅ GIF creation completed!")


if __name__ == "__main__":
    main()
