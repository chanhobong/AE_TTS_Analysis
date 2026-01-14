#!/usr/bin/env python3
"""
evaluate_recon_error.py
- Evaluate reconstruction error as classification score (ROC-AUC, PR-AUC)
- Statistical significance testing (Mann-Whitney U test)
- Bootstrap confidence intervals
- Visualization (boxplot/violin plot)

This is a comprehensive evaluation for the paper!
"""

import os
import sys
import argparse
from typing import Dict, Tuple, List

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import CTROIVolumeDataset

# Import from probe_linear_classifier
import importlib.util
spec = importlib.util.spec_from_file_location("probe_linear_classifier", 
    os.path.join(os.path.dirname(__file__), "probe_linear_classifier.py"))
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)

Tiny3DAE = probe_module.Tiny3DAE
load_model = probe_module.load_model
build_ds = probe_module.build_ds


@torch.no_grad()
def compute_reconstruction_error(model: nn.Module, dataset, device: torch.device, 
                                 batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reconstruction error (MSE) for each sample."""
    model.eval()
    from torch.utils.data import DataLoader
    
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


def compute_auc_metrics(errors: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute ROC-AUC and PR-AUC using reconstruction error as score."""
    # Higher error = more likely to be TTS (anomaly)
    # So we use errors directly as scores
    roc_auc = roc_auc_score(labels, errors)
    pr_auc = average_precision_score(labels, errors)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, errors)
    
    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, errors)
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }


def mann_whitney_test(tts_errors: np.ndarray, normal_errors: np.ndarray) -> Dict:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).
    
    H0: TTS error <= Normal error (TTS errors are not significantly higher)
    H1: TTS error > Normal error (TTS errors are significantly higher)
    """
    statistic, p_value = stats.mannwhitneyu(
        tts_errors, normal_errors, 
        alternative='greater'  # One-sided: TTS > Normal
    )
    
    # Also compute two-sided for completeness
    statistic_two, p_value_two = stats.mannwhitneyu(
        tts_errors, normal_errors, 
        alternative='two-sided'
    )
    
    return {
        "statistic": statistic,
        "p_value_one_sided": p_value,
        "p_value_two_sided": p_value_two,
        "significant_one_sided": p_value < 0.05,
        "significant_two_sided": p_value_two < 0.05,
        "tts_median": np.median(tts_errors),
        "normal_median": np.median(normal_errors),
        "tts_mean": np.mean(tts_errors),
        "normal_mean": np.mean(normal_errors),
    }


def bootstrap_auc(errors: np.ndarray, labels: np.ndarray, n_bootstrap: int = 2000, 
                  confidence: float = 0.95) -> Dict:
    """Bootstrap confidence intervals for AUC metrics."""
    n_samples = len(errors)
    roc_aucs = []
    pr_aucs = []
    ratios = []
    
    tts_mask = labels == 1
    normal_mask = labels == 0
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        errors_boot = errors[indices]
        labels_boot = labels[indices]
        
        # Compute AUCs
        try:
            roc_auc = roc_auc_score(labels_boot, errors_boot)
            pr_auc = average_precision_score(labels_boot, errors_boot)
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)
        except:
            pass  # Skip if all labels are same class
        
        # Compute TTS/Normal ratio
        tts_boot = errors_boot[labels_boot == 1]
        normal_boot = errors_boot[labels_boot == 0]
        if len(tts_boot) > 0 and len(normal_boot) > 0:
            ratio = np.mean(tts_boot) / np.mean(normal_boot)
            ratios.append(ratio)
    
    # Compute confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    roc_auc_ci = np.percentile(roc_aucs, [lower_percentile, upper_percentile])
    pr_auc_ci = np.percentile(pr_aucs, [lower_percentile, upper_percentile])
    ratio_ci = np.percentile(ratios, [lower_percentile, upper_percentile])
    
    return {
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_ci": roc_auc_ci,
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_ci": pr_auc_ci,
        "ratio_mean": np.mean(ratios),
        "ratio_ci": ratio_ci,
        "roc_aucs": roc_aucs,
        "pr_aucs": pr_aucs,
        "ratios": ratios,
    }


def plot_error_distributions(errors_normal_only: np.ndarray, labels_normal_only: np.ndarray,
                             errors_mixed: np.ndarray, labels_mixed: np.ndarray,
                             out_path: str):
    """Plot boxplot/violin plot comparing error distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    data_normal_only = {
        "Normal": errors_normal_only[labels_normal_only == 0],
        "TTS": errors_normal_only[labels_normal_only == 1],
    }
    
    data_mixed = {
        "Normal": errors_mixed[labels_mixed == 0],
        "TTS": errors_mixed[labels_mixed == 1],
    }
    
    # Normal-only model
    ax1 = axes[0]
    positions = [1, 2]
    bp1 = ax1.boxplot([data_normal_only["Normal"], data_normal_only["TTS"]], 
                      positions=positions, widths=0.6, patch_artist=True,
                      labels=["Normal", "TTS"], showmeans=True)
    
    # Color boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    ax1.set_title('Normal-only Model', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    normal_mean = np.mean(data_normal_only["Normal"])
    tts_mean = np.mean(data_normal_only["TTS"])
    ax1.text(0.5, 0.95, f'Normal: {normal_mean:.4f}\nTTS: {tts_mean:.4f}\nRatio: {tts_mean/normal_mean:.2f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    # Mixed model
    ax2 = axes[1]
    bp2 = ax2.boxplot([data_mixed["Normal"], data_mixed["TTS"]], 
                      positions=positions, widths=0.6, patch_artist=True,
                      labels=["Normal", "TTS"], showmeans=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    ax2.set_title('Mixed Model', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    normal_mean = np.mean(data_mixed["Normal"])
    tts_mean = np.mean(data_mixed["TTS"])
    ax2.text(0.5, 0.95, f'Normal: {normal_mean:.4f}\nTTS: {tts_mean:.4f}\nRatio: {tts_mean/normal_mean:.2f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.suptitle('Reconstruction Error Distributions (Test Set)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved distribution plot: {out_path}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_normal", type=str, required=True,
                   help="Path to Normal-only trained model")
    p.add_argument("--model_mixed", type=str, required=True,
                   help="Path to Mixed (TTS+Normal) trained model")
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True, help="Test set CSV")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_bootstrap", type=int, default=2000, help="Number of bootstrap iterations")
    p.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs/recon_error_eval")
    return p


def main():
    args = build_argparser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("RECONSTRUCTION ERROR EVALUATION")
    print("=" * 60)
    print(f"Normal model: {args.model_normal}")
    print(f"Mixed model:  {args.model_mixed}")
    print(f"Test CSV:     {args.test_csv}")
    print(f"Bootstrap:    {args.n_bootstrap} iterations")
    print("=" * 60)
    
    # Build test dataset
    print("\n--- Building Test Dataset ---")
    ds_test_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz,
                           include_csv=args.test_csv, name="TTS(test)")
    ds_test_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                           include_csv=args.test_csv, name="Normal(test)")
    ds_test = ConcatDataset([ds_test_tts, ds_test_norm])
    
    print(f"✅ Test size: {len(ds_test)} (TTS: {len(ds_test_tts)}, Normal: {len(ds_test_norm)})")
    
    # Load models
    print("\n--- Loading Models ---")
    model_normal = load_model(args.model_normal, device)
    model_mixed = load_model(args.model_mixed, device)
    
    # Compute reconstruction errors
    print("\n--- Computing Reconstruction Errors ---")
    print("Normal-only model...")
    errors_normal_only, labels_normal_only = compute_reconstruction_error(
        model_normal, ds_test, device, args.batch_size)
    
    print("Mixed model...")
    errors_mixed, labels_mixed = compute_reconstruction_error(
        model_mixed, ds_test, device, args.batch_size)
    
    # 1. AUC Metrics
    print("\n" + "=" * 60)
    print("1️⃣ RECONSTRUCTION ERROR AS CLASSIFICATION SCORE")
    print("=" * 60)
    
    print("\n--- Normal-only Model ---")
    metrics_normal = compute_auc_metrics(errors_normal_only, labels_normal_only)
    print(f"ROC-AUC: {metrics_normal['roc_auc']:.4f}")
    print(f"PR-AUC:  {metrics_normal['pr_auc']:.4f}")
    
    print("\n--- Mixed Model ---")
    metrics_mixed = compute_auc_metrics(errors_mixed, labels_mixed)
    print(f"ROC-AUC: {metrics_mixed['roc_auc']:.4f}")
    print(f"PR-AUC:  {metrics_mixed['pr_auc']:.4f}")
    
    # 2. Statistical Significance
    print("\n" + "=" * 60)
    print("2️⃣ STATISTICAL SIGNIFICANCE (Mann-Whitney U Test)")
    print("=" * 60)
    
    tts_errors_normal = errors_normal_only[labels_normal_only == 1]
    normal_errors_normal = errors_normal_only[labels_normal_only == 0]
    
    tts_errors_mixed = errors_mixed[labels_mixed == 1]
    normal_errors_mixed = errors_mixed[labels_mixed == 0]
    
    print("\n--- Normal-only Model ---")
    mw_normal = mann_whitney_test(tts_errors_normal, normal_errors_normal)
    print(f"H0: TTS error <= Normal error")
    print(f"H1: TTS error > Normal error")
    print(f"U-statistic: {mw_normal['statistic']:.2f}")
    print(f"p-value (one-sided): {mw_normal['p_value_one_sided']:.6f}")
    print(f"p-value (two-sided): {mw_normal['p_value_two_sided']:.6f}")
    print(f"Significant (one-sided, α=0.05): {mw_normal['significant_one_sided']}")
    print(f"TTS median: {mw_normal['tts_median']:.6f}, Normal median: {mw_normal['normal_median']:.6f}")
    print(f"TTS mean: {mw_normal['tts_mean']:.6f}, Normal mean: {mw_normal['normal_mean']:.6f}")
    
    print("\n--- Mixed Model ---")
    mw_mixed = mann_whitney_test(tts_errors_mixed, normal_errors_mixed)
    print(f"U-statistic: {mw_mixed['statistic']:.2f}")
    print(f"p-value (one-sided): {mw_mixed['p_value_one_sided']:.6f}")
    print(f"p-value (two-sided): {mw_mixed['p_value_two_sided']:.6f}")
    print(f"Significant (one-sided, α=0.05): {mw_mixed['significant_one_sided']}")
    print(f"TTS median: {mw_mixed['tts_median']:.6f}, Normal median: {mw_mixed['normal_median']:.6f}")
    print(f"TTS mean: {mw_mixed['tts_mean']:.6f}, Normal mean: {mw_mixed['normal_mean']:.6f}")
    
    # 3. Bootstrap Confidence Intervals
    print("\n" + "=" * 60)
    print(f"3️⃣ BOOTSTRAP CONFIDENCE INTERVALS ({args.n_bootstrap} iterations)")
    print("=" * 60)
    
    print("\n--- Normal-only Model ---")
    boot_normal = bootstrap_auc(errors_normal_only, labels_normal_only, 
                               args.n_bootstrap, args.confidence)
    print(f"ROC-AUC: {boot_normal['roc_auc_mean']:.4f} "
          f"({args.confidence*100:.0f}% CI: [{boot_normal['roc_auc_ci'][0]:.4f}, {boot_normal['roc_auc_ci'][1]:.4f}])")
    print(f"PR-AUC:  {boot_normal['pr_auc_mean']:.4f} "
          f"({args.confidence*100:.0f}% CI: [{boot_normal['pr_auc_ci'][0]:.4f}, {boot_normal['pr_auc_ci'][1]:.4f}])")
    print(f"TTS/Normal Ratio: {boot_normal['ratio_mean']:.3f} "
          f"({args.confidence*100:.0f}% CI: [{boot_normal['ratio_ci'][0]:.3f}, {boot_normal['ratio_ci'][1]:.3f}])")
    
    print("\n--- Mixed Model ---")
    boot_mixed = bootstrap_auc(errors_mixed, labels_mixed, 
                              args.n_bootstrap, args.confidence)
    print(f"ROC-AUC: {boot_mixed['roc_auc_mean']:.4f} "
          f"({args.confidence*100:.0f}% CI: [{boot_mixed['roc_auc_ci'][0]:.4f}, {boot_mixed['roc_auc_ci'][1]:.4f}])")
    print(f"PR-AUC:  {boot_mixed['pr_auc_mean']:.4f} "
          f"({args.confidence*100:.0f}% CI: [{boot_mixed['pr_auc_ci'][0]:.4f}, {boot_mixed['pr_auc_ci'][1]:.4f}])")
    print(f"TTS/Normal Ratio: {boot_mixed['ratio_mean']:.3f} "
          f"({args.confidence*100:.0f}% CI: [{boot_mixed['ratio_ci'][0]:.3f}, {boot_mixed['ratio_ci'][1]:.3f}])")
    
    # 4. Visualization
    print("\n" + "=" * 60)
    print("4️⃣ VISUALIZATION")
    print("=" * 60)
    
    plot_path = os.path.join(args.out_dir, "error_distributions.png")
    plot_error_distributions(errors_normal_only, labels_normal_only,
                            errors_mixed, labels_mixed, plot_path)
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    
    summary_path = os.path.join(args.out_dir, "recon_error_evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RECONSTRUCTION ERROR EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1️⃣ AUC METRICS\n")
        f.write("-" * 60 + "\n")
        f.write("Normal-only Model:\n")
        f.write(f"  ROC-AUC: {metrics_normal['roc_auc']:.4f}\n")
        f.write(f"  PR-AUC:  {metrics_normal['pr_auc']:.4f}\n\n")
        f.write("Mixed Model:\n")
        f.write(f"  ROC-AUC: {metrics_mixed['roc_auc']:.4f}\n")
        f.write(f"  PR-AUC:  {metrics_mixed['pr_auc']:.4f}\n\n")
        
        f.write("2️⃣ STATISTICAL SIGNIFICANCE (Mann-Whitney U Test)\n")
        f.write("-" * 60 + "\n")
        f.write("Normal-only Model:\n")
        f.write(f"  U-statistic: {mw_normal['statistic']:.2f}\n")
        f.write(f"  p-value (one-sided): {mw_normal['p_value_one_sided']:.6f}\n")
        f.write(f"  p-value (two-sided): {mw_normal['p_value_two_sided']:.6f}\n")
        f.write(f"  Significant (one-sided, α=0.05): {mw_normal['significant_one_sided']}\n")
        f.write(f"  TTS mean: {mw_normal['tts_mean']:.6f}, Normal mean: {mw_normal['normal_mean']:.6f}\n\n")
        f.write("Mixed Model:\n")
        f.write(f"  U-statistic: {mw_mixed['statistic']:.2f}\n")
        f.write(f"  p-value (one-sided): {mw_mixed['p_value_one_sided']:.6f}\n")
        f.write(f"  p-value (two-sided): {mw_mixed['p_value_two_sided']:.6f}\n")
        f.write(f"  Significant (one-sided, α=0.05): {mw_mixed['significant_one_sided']}\n")
        f.write(f"  TTS mean: {mw_mixed['tts_mean']:.6f}, Normal mean: {mw_mixed['normal_mean']:.6f}\n\n")
        
        f.write(f"3️⃣ BOOTSTRAP CONFIDENCE INTERVALS ({args.n_bootstrap} iterations)\n")
        f.write("-" * 60 + "\n")
        f.write("Normal-only Model:\n")
        f.write(f"  ROC-AUC: {boot_normal['roc_auc_mean']:.4f} "
               f"({args.confidence*100:.0f}% CI: [{boot_normal['roc_auc_ci'][0]:.4f}, {boot_normal['roc_auc_ci'][1]:.4f}])\n")
        f.write(f"  PR-AUC:  {boot_normal['pr_auc_mean']:.4f} "
               f"({args.confidence*100:.0f}% CI: [{boot_normal['pr_auc_ci'][0]:.4f}, {boot_normal['pr_auc_ci'][1]:.4f}])\n")
        f.write(f"  TTS/Normal Ratio: {boot_normal['ratio_mean']:.3f} "
               f"({args.confidence*100:.0f}% CI: [{boot_normal['ratio_ci'][0]:.3f}, {boot_normal['ratio_ci'][1]:.3f}])\n\n")
        f.write("Mixed Model:\n")
        f.write(f"  ROC-AUC: {boot_mixed['roc_auc_mean']:.4f} "
               f"({args.confidence*100:.0f}% CI: [{boot_mixed['roc_auc_ci'][0]:.4f}, {boot_mixed['roc_auc_ci'][1]:.4f}])\n")
        f.write(f"  PR-AUC:  {boot_mixed['pr_auc_mean']:.4f} "
               f"({args.confidence*100:.0f}% CI: [{boot_mixed['pr_auc_ci'][0]:.4f}, {boot_mixed['pr_auc_ci'][1]:.4f}])\n")
        f.write(f"  TTS/Normal Ratio: {boot_mixed['ratio_mean']:.3f} "
               f"({args.confidence*100:.0f}% CI: [{boot_mixed['ratio_ci'][0]:.3f}, {boot_mixed['ratio_ci'][1]:.3f}])\n")
    
    print(f"\n💾 Results saved to: {summary_path}")
    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main()
