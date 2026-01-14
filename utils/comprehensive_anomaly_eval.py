#!/usr/bin/env python3
"""
comprehensive_anomaly_eval.py
- Comprehensive anomaly detection evaluation across all models
- Parameter tuning on val set
- Final evaluation on test set (1 time only)
- Bootstrap CI, Mann-Whitney U test
- Comparison table

Models:
1. AE Normal-only (reconstruction error)
2. AE Mixed (reconstruction error)
3. DDPM (anomaly score with T_small, K)
4. Flow-Matching (anomaly score with K, t-range)
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from itertools import product

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import pandas as pd

from data.dataset import CTROIVolumeDataset
from models import LatentVelocityEstimator
from models.ddpm import NoiseSchedule, compute_anomaly_score_ddpm, compute_anomaly_score_ddpm_small_t
from models.flow_matching import compute_anomaly_score_mse_multi_t
from latent_extraction import LatentDataset

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
def compute_ae_recon_error(model: nn.Module, dataset, device: torch.device, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AE reconstruction error (MSE)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_errors = []
    all_labels = []
    
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        
        x_hat = model(x)
        mse = nn.functional.mse_loss(x_hat, x, reduction='none')
        mse_per_sample = mse.view(x.size(0), -1).mean(dim=1).cpu().numpy()
        
        all_errors.append(mse_per_sample)
        all_labels.append(y)
    
    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels)
    
    return errors, labels


@torch.no_grad()
def compute_ddpm_scores(
    model: nn.Module,
    latent_dataset: LatentDataset,
    noise_schedule: NoiseSchedule,
    t_small: Optional[int],
    K: int,
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DDPM anomaly scores."""
    model.eval()
    loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_scores = []
    
    # Get labels from original dataset
    labels = []
    for idx in range(len(latent_dataset.volume_dataset)):
        sample = latent_dataset.volume_dataset[idx]
        labels.append(sample["y"])
    labels = np.array(labels)
    
    for batch_idx, batch in enumerate(loader):
        z_batch = batch["z"].to(device)  # (B, C, D, H, W)
        
        # Compute scores for each sample in batch
        batch_scores = []
        for i in range(z_batch.shape[0]):
            z_single = z_batch[i:i+1]  # (1, C, D, H, W)
            
            if t_small is not None:
                score = compute_anomaly_score_ddpm_small_t(
                    model, z_single, noise_schedule, t_max=t_small, n_samples=K, device=device
                )
            else:
                score = compute_anomaly_score_ddpm(
                    model, z_single, noise_schedule, n_samples=K, device=device
                )
            
            batch_scores.append(score.item())
        
        all_scores.extend(batch_scores)
    
    scores = np.array(all_scores)
    
    return scores, labels


@torch.no_grad()
def compute_fm_scores(
    model: nn.Module,
    latent_dataset: LatentDataset,
    K: int,
    t_range: Optional[Tuple[float, float]],
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Flow-Matching anomaly scores."""
    model.eval()
    loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_scores = []
    
    # Get labels from original dataset
    labels = []
    for idx in range(len(latent_dataset.volume_dataset)):
        sample = latent_dataset.volume_dataset[idx]
        labels.append(sample["y"])
    labels = np.array(labels)
    
    for batch_idx, batch in enumerate(loader):
        z_batch = batch["z"].to(device)  # (B, C, D, H, W)
        
        # Compute scores for each sample in batch
        batch_scores = []
        for i in range(z_batch.shape[0]):
            z_single = z_batch[i:i+1]  # (1, C, D, H, W)
            
            # Use modified version if t_range is specified
            if t_range is not None:
                # Sample t from specified range
                t_min, t_max = t_range
                scores_list = []
                for _ in range(K):
                    t = torch.rand(1, device=device) * (t_max - t_min) + t_min
                    z_1 = torch.randn_like(z_single)
                    t_expanded = t[:, None, None, None, None]
                    z_t = (1 - t_expanded) * z_single + t_expanded * z_1
                    v_pred = model(z_t, t)
                    v_true = z_1 - z_single
                    error = nn.functional.mse_loss(v_pred, v_true, reduction='none')
                    error = error.mean(dim=(1, 2, 3, 4))
                    scores_list.append(error)
                score = torch.stack(scores_list).mean()
            else:
                score = compute_anomaly_score_mse_multi_t(
                    model, z_single, n_samples=K, device=device
                )
            
            batch_scores.append(score.item())
        
        all_scores.extend(batch_scores)
    
    scores = np.array(all_scores)
    
    return scores, labels


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute ROC-AUC, PR-AUC, and Mann-Whitney U test."""
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Mann-Whitney U test (one-sided: TTS > Normal)
    tts_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    
    if len(tts_scores) > 0 and len(normal_scores) > 0:
        statistic, p_value = stats.mannwhitneyu(
            tts_scores, normal_scores, alternative='greater'
        )
        significant = p_value < 0.05
    else:
        statistic, p_value, significant = None, None, None
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mw_statistic": statistic,
        "mw_p_value": p_value,
        "mw_significant": significant,
        "tts_mean": tts_scores.mean() if len(tts_scores) > 0 else None,
        "normal_mean": normal_scores.mean() if len(normal_scores) > 0 else None,
    }


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n_bootstrap: int = 2000, confidence: float = 0.95) -> Dict:
    """Bootstrap confidence intervals."""
    n_samples = len(scores)
    roc_aucs = []
    pr_aucs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        scores_boot = scores[indices]
        labels_boot = labels[indices]
        
        try:
            roc_auc = roc_auc_score(labels_boot, scores_boot)
            pr_auc = average_precision_score(labels_boot, scores_boot)
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)
        except:
            pass
    
    alpha = 1 - confidence
    lower = (alpha / 2) * 100
    upper = (1 - alpha / 2) * 100
    
    return {
        "roc_auc_ci": np.percentile(roc_aucs, [lower, upper]) if roc_aucs else None,
        "pr_auc_ci": np.percentile(pr_aucs, [lower, upper]) if pr_aucs else None,
    }


def tune_parameters_val(
    model_type: str,
    model: nn.Module,
    latent_dataset_val: LatentDataset,
    labels_val: np.ndarray,
    device: torch.device,
    param_grid: Dict,
    noise_schedule: Optional[NoiseSchedule] = None,
    batch_size: int = 8,
) -> Dict:
    """Tune parameters on validation set."""
    print(f"\n--- Tuning {model_type} parameters on val set ---")
    
    best_params = None
    best_auc = -1.0
    all_results = []
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)
    print(f"  Total combinations to try: {total_combinations}")
    
    for idx, params in enumerate(product(*param_values), 1):
        param_dict = dict(zip(param_names, params))
        
        # Compute scores
        if model_type == "DDPM":
            t_small = param_dict.get("t_small")
            K = param_dict["K"]
            scores, _ = compute_ddpm_scores(
                model, latent_dataset_val, noise_schedule, t_small, K, device, batch_size
            )
        elif model_type == "FM":
            K = param_dict["K"]
            t_range = param_dict.get("t_range")
            scores, _ = compute_fm_scores(
                model, latent_dataset_val, K, t_range, device, batch_size
            )
        else:
            continue
        
        # Compute metrics
        metrics = compute_metrics(scores, labels_val)
        auc = metrics["roc_auc"]
        
        # Check direction: TTS should have higher scores
        tts_mean = metrics["tts_mean"]
        normal_mean = metrics["normal_mean"]
        direction_ok = (tts_mean is not None and normal_mean is not None and tts_mean > normal_mean)
        
        all_results.append({
            **param_dict, 
            "roc_auc": auc, 
            "pr_auc": metrics["pr_auc"],
            "tts_mean": tts_mean,
            "normal_mean": normal_mean,
            "direction_ok": direction_ok,
        })
        
        direction_str = "✓" if direction_ok else "✗"
        print(f"  [{idx}/{total_combinations}] Params: {param_dict} | ROC-AUC: {auc:.4f} | Direction: {direction_str}")
        
        # Only consider if direction is correct
        if direction_ok and auc > best_auc:
            best_auc = auc
            best_params = param_dict
    
    if best_params is None:
        print("⚠️  Warning: No valid parameters found (TTS scores not higher than Normal). Using best AUC regardless.")
        best_params = max(all_results, key=lambda x: x["roc_auc"])
        best_params = {k: best_params[k] for k in param_names}
        best_auc = max(all_results, key=lambda x: x["roc_auc"])["roc_auc"]
    
    print(f"✅ Best params: {best_params} (ROC-AUC: {best_auc:.4f})")
    
    return {
        "best_params": best_params,
        "best_auc": best_auc,
        "all_results": all_results,
    }


def build_argparser():
    p = argparse.ArgumentParser()
    
    # Model paths
    p.add_argument("--ae_normal_path", type=str, required=True, help="Normal-only AE model path")
    p.add_argument("--ae_mixed_path", type=str, required=True, help="Mixed AE model path")
    p.add_argument("--ddpm_path", type=str, required=True, help="DDPM model path")
    p.add_argument("--fm_path", type=str, required=True, help="Flow-Matching model path")
    
    # Data paths
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    
    # Parameter grids for tuning
    p.add_argument("--ddpm_t_small", type=int, nargs="+", default=[10, 20, 50, 100],
                   help="T_small values to try (use --ddpm_use_full_range to also try full range)")
    p.add_argument("--ddpm_use_full_range", action="store_true",
                   help="Also try full timestep range (None) for DDPM")
    p.add_argument("--ddpm_K", type=int, nargs="+", default=[5, 10, 20], help="K values for DDPM")
    p.add_argument("--fm_K", type=int, nargs="+", default=[5, 10, 20], help="K values for FM")
    p.add_argument("--fm_t_ranges", type=str, nargs="+", default=["full", "small"],
                   help="t-range options: 'full'=[0,1], 'small'=[0,0.3]")
    
    # Evaluation
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs/comprehensive_eval")
    
    return p


def main():
    args = build_argparser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("COMPREHENSIVE ANOMALY DETECTION EVALUATION")
    print("=" * 60)
    
    # Build datasets
    print("\n--- Building Datasets ---")
    ds_val_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz,
                         include_csv=args.val_csv, name="TTS(val)")
    ds_val_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                          include_csv=args.val_csv, name="Normal(val)")
    ds_val = ConcatDataset([ds_val_tts, ds_val_norm])
    
    ds_test_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz,
                          include_csv=args.test_csv, name="TTS(test)")
    ds_test_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                           include_csv=args.test_csv, name="Normal(test)")
    ds_test = ConcatDataset([ds_test_tts, ds_test_norm])
    
    # Get labels
    labels_val = np.array([ds_val[i]["y"] for i in range(len(ds_val))])
    labels_test = np.array([ds_test[i]["y"] for i in range(len(ds_test))])
    
    print(f"✅ Val size: {len(ds_val)} (TTS: {sum(labels_val==1)}, Normal: {sum(labels_val==0)})")
    print(f"✅ Test size: {len(ds_test)} (TTS: {sum(labels_test==1)}, Normal: {sum(labels_test==0)})")
    
    results = {}
    
    # 1. AE Normal-only
    print("\n" + "=" * 60)
    print("1. AE NORMAL-ONLY")
    print("=" * 60)
    model_ae_normal = load_model(args.ae_normal_path, device)
    scores_ae_normal_val, _ = compute_ae_recon_error(model_ae_normal, ds_val, device, args.batch_size)
    scores_ae_normal_test, _ = compute_ae_recon_error(model_ae_normal, ds_test, device, args.batch_size)
    
    metrics_val = compute_metrics(scores_ae_normal_val, labels_val)
    metrics_test = compute_metrics(scores_ae_normal_test, labels_test)
    ci_test = bootstrap_ci(scores_ae_normal_test, labels_test, args.n_bootstrap, args.confidence)
    
    results["AE_Normal-only"] = {
        "params": {},
        "val_roc_auc": metrics_val["roc_auc"],
        "test_roc_auc": metrics_test["roc_auc"],
        "test_pr_auc": metrics_test["pr_auc"],
        "test_roc_auc_ci": ci_test["roc_auc_ci"],
        "test_pr_auc_ci": ci_test["pr_auc_ci"],
        "test_mw_p_value": metrics_test["mw_p_value"],
        "test_mw_significant": metrics_test["mw_significant"],
    }
    
    # 2. AE Mixed
    print("\n" + "=" * 60)
    print("2. AE MIXED")
    print("=" * 60)
    model_ae_mixed = load_model(args.ae_mixed_path, device)
    scores_ae_mixed_val, _ = compute_ae_recon_error(model_ae_mixed, ds_val, device, args.batch_size)
    scores_ae_mixed_test, _ = compute_ae_recon_error(model_ae_mixed, ds_test, device, args.batch_size)
    
    metrics_val = compute_metrics(scores_ae_mixed_val, labels_val)
    metrics_test = compute_metrics(scores_ae_mixed_test, labels_test)
    ci_test = bootstrap_ci(scores_ae_mixed_test, labels_test, args.n_bootstrap, args.confidence)
    
    results["AE_Mixed"] = {
        "params": {},
        "val_roc_auc": metrics_val["roc_auc"],
        "test_roc_auc": metrics_test["roc_auc"],
        "test_pr_auc": metrics_test["pr_auc"],
        "test_roc_auc_ci": ci_test["roc_auc_ci"],
        "test_pr_auc_ci": ci_test["pr_auc_ci"],
        "test_mw_p_value": metrics_test["mw_p_value"],
        "test_mw_significant": metrics_test["mw_significant"],
    }
    
    # 3. DDPM - Tune parameters on val
    print("\n" + "=" * 60)
    print("3. DDPM - PARAMETER TUNING")
    print("=" * 60)
    
    # Load DDPM model and noise schedule
    checkpoint_ddpm = torch.load(args.ddpm_path, map_location=device)
    model_ddpm = LatentVelocityEstimator(
        in_channels=128,  # Should match AE encoder output
        base_channels=64,
        time_dim=128,
        num_res_blocks=4,
    ).to(device)
    model_ddpm.load_state_dict(checkpoint_ddpm["model"])
    
    noise_schedule_info = checkpoint_ddpm.get("noise_schedule", {"timesteps": 1000, "schedule_type": "cosine"})
    noise_schedule = NoiseSchedule(
        timesteps=noise_schedule_info["timesteps"],
        schedule_type=noise_schedule_info["schedule_type"],
        device=device
    )
    
    # Extract latents for val and test
    latent_ds_val = LatentDataset(model_ae_normal, ds_val, device)
    latent_ds_test = LatentDataset(model_ae_normal, ds_test, device)
    
    # Parameter grid
    t_small_values = list(args.ddpm_t_small)
    if args.ddpm_use_full_range:
        t_small_values.append(None)  # None means full range
    
    param_grid_ddpm = {
        "t_small": t_small_values,
        "K": args.ddpm_K,
    }
    
    # Tune on val
    tune_results_ddpm = tune_parameters_val(
        "DDPM", model_ddpm, latent_ds_val, labels_val, device, param_grid_ddpm, noise_schedule
    )
    best_params_ddpm = tune_results_ddpm["best_params"]
    
    # Evaluate on test with best params
    print(f"\n--- Evaluating DDPM on test with best params: {best_params_ddpm} ---")
    scores_ddpm_test, _ = compute_ddpm_scores(
        model_ddpm, latent_ds_test, noise_schedule,
        best_params_ddpm["t_small"], best_params_ddpm["K"], device, args.batch_size
    )
    
    metrics_test = compute_metrics(scores_ddpm_test, labels_test)
    ci_test = bootstrap_ci(scores_ddpm_test, labels_test, args.n_bootstrap, args.confidence)
    
    results["DDPM"] = {
        "params": best_params_ddpm,
        "val_roc_auc": tune_results_ddpm["best_auc"],
        "test_roc_auc": metrics_test["roc_auc"],
        "test_pr_auc": metrics_test["pr_auc"],
        "test_roc_auc_ci": ci_test["roc_auc_ci"],
        "test_pr_auc_ci": ci_test["pr_auc_ci"],
        "test_mw_p_value": metrics_test["mw_p_value"],
        "test_mw_significant": metrics_test["mw_significant"],
    }
    
    # 4. Flow-Matching - Tune parameters on val
    print("\n" + "=" * 60)
    print("4. FLOW-MATCHING - PARAMETER TUNING")
    print("=" * 60)
    
    # Load FM model
    checkpoint_fm = torch.load(args.fm_path, map_location=device)
    model_fm = LatentVelocityEstimator(
        in_channels=128,
        base_channels=64,
        time_dim=128,
        num_res_blocks=4,
    ).to(device)
    model_fm.load_state_dict(checkpoint_fm["model"])
    
    # Parameter grid
    t_range_map = {"full": None, "small": (0.0, 0.3)}
    t_ranges = [t_range_map.get(tr, None) for tr in args.fm_t_ranges]
    
    param_grid_fm = {
        "K": args.fm_K,
        "t_range": t_ranges,
    }
    
    # Tune on val
    tune_results_fm = tune_parameters_val(
        "FM", model_fm, latent_ds_val, labels_val, device, param_grid_fm
    )
    best_params_fm = tune_results_fm["best_params"]
    
    # Evaluate on test with best params
    print(f"\n--- Evaluating FM on test with best params: {best_params_fm} ---")
    scores_fm_test, _ = compute_fm_scores(
        model_fm, latent_ds_test, best_params_fm["K"], best_params_fm["t_range"], device, args.batch_size
    )
    
    metrics_test = compute_metrics(scores_fm_test, labels_test)
    ci_test = bootstrap_ci(scores_fm_test, labels_test, args.n_bootstrap, args.confidence)
    
    results["Flow-Matching"] = {
        "params": best_params_fm,
        "val_roc_auc": tune_results_fm["best_auc"],
        "test_roc_auc": metrics_test["roc_auc"],
        "test_pr_auc": metrics_test["pr_auc"],
        "test_roc_auc_ci": ci_test["roc_auc_ci"],
        "test_pr_auc_ci": ci_test["pr_auc_ci"],
        "test_mw_p_value": metrics_test["mw_p_value"],
        "test_mw_significant": metrics_test["mw_significant"],
    }
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create comparison table
    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE")
    print("=" * 60)
    
    table_data = []
    for model_name, result in results.items():
        params_str = str(result["params"]) if result["params"] else "N/A"
        roc_ci = result["test_roc_auc_ci"]
        pr_ci = result["test_pr_auc_ci"]
        
        table_data.append({
            "Model": model_name,
            "Params": params_str,
            "Val ROC-AUC": f"{result['val_roc_auc']:.4f}",
            "Test ROC-AUC": f"{result['test_roc_auc']:.4f}",
            "ROC-AUC CI (95%)": f"[{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]" if roc_ci is not None else "N/A",
            "Test PR-AUC": f"{result['test_pr_auc']:.4f}",
            "PR-AUC CI (95%)": f"[{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]" if pr_ci is not None else "N/A",
            "MW p-value": f"{result['test_mw_p_value']:.6f}" if result['test_mw_p_value'] is not None else "N/A",
            "MW Significant": "Yes" if result['test_mw_significant'] else "No",
        })
    
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    
    # Save to files
    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved comparison table: {csv_path}")
    
    # Save detailed summary
    summary_path = os.path.join(args.out_dir, "detailed_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("COMPREHENSIVE ANOMALY DETECTION EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Params: {result['params']}\n")
            f.write(f"  Val ROC-AUC: {result['val_roc_auc']:.4f}\n")
            f.write(f"  Test ROC-AUC: {result['test_roc_auc']:.4f}\n")
            if result['test_roc_auc_ci'] is not None:
                f.write(f"  Test ROC-AUC CI (95%): [{result['test_roc_auc_ci'][0]:.4f}, {result['test_roc_auc_ci'][1]:.4f}]\n")
            f.write(f"  Test PR-AUC: {result['test_pr_auc']:.4f}\n")
            if result['test_pr_auc_ci'] is not None:
                f.write(f"  Test PR-AUC CI (95%): [{result['test_pr_auc_ci'][0]:.4f}, {result['test_pr_auc_ci'][1]:.4f}]\n")
            f.write(f"  Mann-Whitney U p-value: {result['test_mw_p_value']:.6f}\n")
            f.write(f"  Significant (α=0.05): {result['test_mw_significant']}\n")
            f.write("\n")
    
    print(f"💾 Saved detailed summary: {summary_path}")
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
