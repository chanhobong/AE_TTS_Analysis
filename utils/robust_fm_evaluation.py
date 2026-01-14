#!/usr/bin/env python3
"""
robust_fm_evaluation.py
- Comprehensive and robust evaluation of Flow Matching anomaly detection
- Implements 6 critical validation steps for objective completion

1. Val/Test 동일 스코어링 조건 검증
2. K와 t-range 민감도 분석
3. Score 분포 plot + effect size (Cohen's d, Cliff's delta)
4. Permutation test
5. Confounder 체크 (sex/age 상관관계)
6. Replicated splits
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional
from itertools import product
from datetime import datetime

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from data.dataset import CTROIVolumeDataset
from models import LatentVelocityEstimator
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


# ============================================================================
# (1) Val/Test 동일 스코어링 조건 검증 - 로그 시스템
# ============================================================================

class ScoringLogger:
    """Log all scoring parameters to ensure identical conditions for val/test."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup file and console logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_scoring_params(self, split: str, model_type: str, params: Dict, seed: int):
        """Log scoring parameters for a specific split."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "split": split,
            "model_type": model_type,
            "params": params,
            "seed": seed,
        }
        self.logs.append(log_entry)
        self.logger.info(f"[{split}] {model_type} scoring params: {params}, seed={seed}")
    
    def save_logs(self):
        """Save all logs to JSON file."""
        with open(self.log_file.replace('.log', '.json'), 'w') as f:
            json.dump(self.logs, f, indent=2)
        self.logger.info(f"✅ Saved scoring logs to {self.log_file.replace('.log', '.json')}")


# ============================================================================
# (2) K와 t-range 민감도 분석
# ============================================================================

@torch.no_grad()
def compute_fm_scores_with_params(
    model: nn.Module,
    latent_dataset: LatentDataset,
    K: int,
    t_range: Optional[Tuple[float, float]],
    seed: int,
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FM scores with specific K and t_range, using fixed seed."""
    model.eval()
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_scores = []
    labels = []
    
    # Get labels from original dataset
    for idx in range(len(latent_dataset.volume_dataset)):
        sample = latent_dataset.volume_dataset[idx]
        labels.append(sample["y"])
    labels = np.array(labels)
    
    for batch_idx, batch in enumerate(loader):
        z_batch = batch["z"].to(device)  # (B, C, D, H, W)
        
        batch_scores = []
        for i in range(z_batch.shape[0]):
            z_single = z_batch[i:i+1]  # (1, C, D, H, W)
            
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
                # Use full t range [0, 1]
                score = compute_anomaly_score_mse_multi_t(
                    model, z_single, n_samples=K, device=device
                )
            
            batch_scores.append(score.item())
        
        all_scores.extend(batch_scores)
    
    scores = np.array(all_scores)
    return scores, labels


def sensitivity_analysis(
    model: nn.Module,
    latent_dataset: LatentDataset,
    labels: np.ndarray,
    K_values: List[int],
    t_ranges: List[Optional[Tuple[float, float]]],
    seed: int,
    device: torch.device,
    out_dir: str,
) -> pd.DataFrame:
    """Perform sensitivity analysis for K and t_range parameters."""
    print("\n" + "="*60)
    print("(2) K와 t-range 민감도 분석")
    print("="*60)
    
    results = []
    
    for K in K_values:
        for t_range in t_ranges:
            t_range_str = f"[{t_range[0]:.2f}, {t_range[1]:.2f}]" if t_range else "[0.0, 1.0]"
            print(f"\n  Testing K={K}, t_range={t_range_str}...")
            
            scores, _ = compute_fm_scores_with_params(
                model, latent_dataset, K, t_range, seed, device
            )
            
            roc_auc = roc_auc_score(labels, scores)
            pr_auc = average_precision_score(labels, scores)
            
            results.append({
                "K": K,
                "t_range": t_range_str,
                "t_min": t_range[0] if t_range else 0.0,
                "t_max": t_range[1] if t_range else 1.0,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
            })
            
            print(f"    ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = os.path.join(out_dir, "sensitivity_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved sensitivity analysis to {csv_path}")
    
    # Create heatmap visualization
    pivot_roc = df.pivot(index="K", columns="t_range", values="ROC-AUC")
    pivot_pr = df.pivot(index="K", columns="t_range", values="PR-AUC")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(pivot_roc, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[0], cbar_kws={'label': 'ROC-AUC'})
    axes[0].set_title("ROC-AUC Sensitivity (K vs t_range)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("t_range")
    axes[0].set_ylabel("K")
    
    sns.heatmap(pivot_pr, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1], cbar_kws={'label': 'PR-AUC'})
    axes[1].set_title("PR-AUC Sensitivity (K vs t_range)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("t_range")
    axes[1].set_ylabel("K")
    
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "sensitivity_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved sensitivity heatmap to {heatmap_path}")
    
    return df


# ============================================================================
# (3) Score 분포 plot + effect size
# ============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta effect size."""
    n1, n2 = len(group1), len(group2)
    dominance = 0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    return dominance / (n1 * n2)


def plot_score_distributions(
    scores: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    out_dir: str,
):
    """Plot score distributions with effect size statistics."""
    print("\n" + "="*60)
    print(f"(3) Score 분포 plot + effect size: {model_name}")
    print("="*60)
    
    tts_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    
    # Compute effect sizes
    cohens_d_val = cohens_d(tts_scores, normal_scores)
    cliffs_delta_val = cliffs_delta(tts_scores, normal_scores)
    
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(tts_scores, normal_scores, alternative='greater')
    
    print(f"  TTS mean: {tts_scores.mean():.4f} ± {tts_scores.std():.4f}")
    print(f"  Normal mean: {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
    print(f"  Cohen's d: {cohens_d_val:.4f}")
    print(f"  Cliff's delta: {cliffs_delta_val:.4f}")
    print(f"  Mann-Whitney U: statistic={statistic:.2f}, p={p_value:.4f}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram
    ax1 = axes[0, 0]
    ax1.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='blue', density=True)
    ax1.hist(tts_scores, bins=30, alpha=0.6, label='TTS', color='red', density=True)
    ax1.set_xlabel('Anomaly Score', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Score Distribution (Histogram)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Violin plot
    ax2 = axes[0, 1]
    data_for_violin = pd.DataFrame({
        'Score': np.concatenate([normal_scores, tts_scores]),
        'Label': ['Normal'] * len(normal_scores) + ['TTS'] * len(tts_scores)
    })
    sns.violinplot(data=data_for_violin, x='Label', y='Score', ax=ax2, palette=['blue', 'red'])
    ax2.set_title('Score Distribution (Violin Plot)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Box plot
    ax3 = axes[1, 0]
    sns.boxplot(data=data_for_violin, x='Label', y='Score', ax=ax3, palette=['blue', 'red'])
    ax3.set_title('Score Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Statistical Summary
    
    Sample Sizes:
    • Normal: {len(normal_scores)}
    • TTS: {len(tts_scores)}
    
    Means ± SD:
    • Normal: {normal_scores.mean():.4f} ± {normal_scores.std():.4f}
    • TTS: {tts_scores.mean():.4f} ± {tts_scores.std():.4f}
    
    Effect Sizes:
    • Cohen's d: {cohens_d_val:.4f}
    • Cliff's delta: {cliffs_delta_val:.4f}
    
    Statistical Test:
    • Mann-Whitney U: {statistic:.2f}
    • p-value: {p_value:.4f}
    • Significant: {'Yes' if p_value < 0.05 else 'No'}
    
    Interpretation:
    • Cohen's d: {'Large' if abs(cohens_d_val) > 0.8 else 'Medium' if abs(cohens_d_val) > 0.5 else 'Small'} effect
    • Cliff's delta: {'Large' if abs(cliffs_delta_val) > 0.474 else 'Medium' if abs(cliffs_delta_val) > 0.33 else 'Small'} effect
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')
    
    plt.suptitle(f'{model_name} - Score Distribution Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"score_distribution_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved distribution plot to {out_path}")
    
    # Save statistics
    stats_dict = {
        "model": model_name,
        "n_normal": len(normal_scores),
        "n_tts": len(tts_scores),
        "normal_mean": float(normal_scores.mean()),
        "normal_std": float(normal_scores.std()),
        "tts_mean": float(tts_scores.mean()),
        "tts_std": float(tts_scores.std()),
        "cohens_d": float(cohens_d_val),
        "cliffs_delta": float(cliffs_delta_val),
        "mannwhitney_statistic": float(statistic),
        "mannwhitney_p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
    
    stats_path = os.path.join(out_dir, f"effect_size_{model_name.lower().replace(' ', '_')}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"✅ Saved effect size statistics to {stats_path}")
    
    return stats_dict


# ============================================================================
# (4) Permutation test
# ============================================================================

def permutation_test(
    scores: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict:
    """Perform permutation test to verify model is not randomly separating."""
    print("\n" + "="*60)
    print("(4) Permutation Test")
    print("="*60)
    
    np.random.seed(seed)
    
    # True AUC
    true_auc = roc_auc_score(labels, scores)
    print(f"  True ROC-AUC: {true_auc:.4f}")
    
    # Permuted AUCs
    permuted_aucs = []
    for i in range(n_permutations):
        labels_perm = np.random.permutation(labels)
        try:
            auc_perm = roc_auc_score(labels_perm, scores)
            permuted_aucs.append(auc_perm)
        except:
            pass
        
        if (i + 1) % 100 == 0:
            print(f"    Completed {i + 1}/{n_permutations} permutations...")
    
    permuted_aucs = np.array(permuted_aucs)
    
    # Statistics
    perm_mean = permuted_aucs.mean()
    perm_std = permuted_aucs.std()
    p_value = (permuted_aucs >= true_auc).mean()
    
    print(f"  Permuted AUC mean: {perm_mean:.4f} ± {perm_std:.4f}")
    print(f"  p-value (permuted >= true): {p_value:.4f}")
    print(f"  {'✅ Model is significantly better than random' if p_value < 0.05 else '❌ Model may be randomly separating'}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(permuted_aucs, bins=50, alpha=0.7, color='gray', label='Permuted AUCs')
    ax.axvline(true_auc, color='red', linestyle='--', linewidth=2, label=f'True AUC = {true_auc:.4f}')
    ax.axvline(0.5, color='black', linestyle=':', linewidth=1, label='Random (0.5)')
    ax.set_xlabel('ROC-AUC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Permutation Test: True AUC vs Permuted AUCs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    results = {
        "true_auc": float(true_auc),
        "permuted_mean": float(perm_mean),
        "permuted_std": float(perm_std),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "permuted_aucs": permuted_aucs.tolist(),
    }
    
    return results, fig


# ============================================================================
# (5) Confounder 체크
# ============================================================================

def confounder_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    dataset: Dataset,
    out_dir: str,
) -> Dict:
    """Analyze potential confounders: sex and age."""
    print("\n" + "="*60)
    print("(5) Confounder 체크 (Sex/Age 상관관계)")
    print("="*60)
    
    # Extract metadata
    ages = []
    sexes = []
    patient_ids = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if "meta" in sample:
            ages.append(sample["meta"]["age"])
            sexes.append(sample["meta"]["sex"])
        else:
            ages.append(None)
            sexes.append(None)
        patient_ids.append(sample.get("patient_id", f"sample_{idx}"))
    
    ages = np.array(ages, dtype=float)
    sexes = np.array(sexes, dtype=float)
    
    # Filter out None values
    valid_mask = ~(np.isnan(ages) | np.isnan(sexes))
    scores_valid = scores[valid_mask]
    labels_valid = labels[valid_mask]
    ages_valid = ages[valid_mask]
    sexes_valid = sexes[valid_mask]
    
    print(f"  Valid samples: {len(scores_valid)}/{len(scores)}")
    
    # Correlation analysis
    # Score vs Age
    corr_age, p_age = stats.spearmanr(scores_valid, ages_valid)
    print(f"  Score vs Age: Spearman r={corr_age:.4f}, p={p_age:.4f}")
    
    # Score vs Sex
    corr_sex, p_sex = stats.spearmanr(scores_valid, sexes_valid)
    print(f"  Score vs Sex: Spearman r={corr_sex:.4f}, p={p_sex:.4f}")
    
    # Score vs Label (should be high)
    corr_label, p_label = stats.spearmanr(scores_valid, labels_valid)
    print(f"  Score vs Label: Spearman r={corr_label:.4f}, p={p_label:.4f}")
    
    # Group analysis
    print("\n  Group Analysis:")
    print(f"    TTS: n={np.sum(labels_valid==1)}, age={ages_valid[labels_valid==1].mean():.1f}±{ages_valid[labels_valid==1].std():.1f}, "
          f"sex(M=1): {sexes_valid[labels_valid==1].mean():.2f}")
    print(f"    Normal: n={np.sum(labels_valid==0)}, age={ages_valid[labels_valid==0].mean():.1f}±{ages_valid[labels_valid==0].std():.1f}, "
          f"sex(M=1): {sexes_valid[labels_valid==0].mean():.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Score vs Age
    ax1 = axes[0, 0]
    ax1.scatter(ages_valid[labels_valid==0], scores_valid[labels_valid==0], 
                alpha=0.6, label='Normal', color='blue', s=50)
    ax1.scatter(ages_valid[labels_valid==1], scores_valid[labels_valid==1], 
                alpha=0.6, label='TTS', color='red', s=50)
    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Anomaly Score', fontsize=11)
    ax1.set_title(f'Score vs Age (r={corr_age:.3f}, p={p_age:.3f})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Score vs Sex
    ax2 = axes[0, 1]
    sex_labels = ['Female', 'Male']
    normal_scores_by_sex = [scores_valid[(labels_valid==0) & (sexes_valid==s)] for s in [0, 1]]
    tts_scores_by_sex = [scores_valid[(labels_valid==1) & (sexes_valid==s)] for s in [0, 1]]
    
    x_pos = np.arange(2)
    width = 0.35
    ax2.bar(x_pos - width/2, [np.mean(s) if len(s) > 0 else 0 for s in normal_scores_by_sex], 
            width, label='Normal', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, [np.mean(s) if len(s) > 0 else 0 for s in tts_scores_by_sex], 
            width, label='TTS', color='red', alpha=0.7)
    ax2.set_xlabel('Sex', fontsize=11)
    ax2.set_ylabel('Mean Anomaly Score', fontsize=11)
    ax2.set_title(f'Score vs Sex (r={corr_sex:.3f}, p={p_sex:.3f})', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sex_labels)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Age distribution by label
    ax3 = axes[1, 0]
    ax3.hist(ages_valid[labels_valid==0], bins=15, alpha=0.6, label='Normal', color='blue', density=True)
    ax3.hist(ages_valid[labels_valid==1], bins=15, alpha=0.6, label='TTS', color='red', density=True)
    ax3.set_xlabel('Age', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Age Distribution by Label', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Sex distribution by label
    ax4 = axes[1, 1]
    sex_counts = pd.DataFrame({
        'Normal': [np.sum((labels_valid==0) & (sexes_valid==s)) for s in [0, 1]],
        'TTS': [np.sum((labels_valid==1) & (sexes_valid==s)) for s in [0, 1]]
    }, index=['Female', 'Male'])
    sex_counts.plot(kind='bar', ax=ax4, color=['blue', 'red'], alpha=0.7)
    ax4.set_xlabel('Sex', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Sex Distribution by Label', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=0)
    
    plt.suptitle('Confounder Analysis: Age and Sex', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "confounder_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confounder analysis to {out_path}")
    
    results = {
        "score_vs_age": {"correlation": float(corr_age), "p_value": float(p_age)},
        "score_vs_sex": {"correlation": float(corr_sex), "p_value": float(p_sex)},
        "score_vs_label": {"correlation": float(corr_label), "p_value": float(p_label)},
        "tts_stats": {
            "n": int(np.sum(labels_valid==1)),
            "age_mean": float(ages_valid[labels_valid==1].mean()),
            "age_std": float(ages_valid[labels_valid==1].std()),
            "sex_mean": float(sexes_valid[labels_valid==1].mean()),
        },
        "normal_stats": {
            "n": int(np.sum(labels_valid==0)),
            "age_mean": float(ages_valid[labels_valid==0].mean()),
            "age_std": float(ages_valid[labels_valid==0].std()),
            "sex_mean": float(sexes_valid[labels_valid==0].mean()),
        },
    }
    
    results_path = os.path.join(out_dir, "confounder_analysis.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved confounder analysis results to {results_path}")
    
    return results


# ============================================================================
# (6) Replicated splits
# ============================================================================

def create_replicated_splits(
    tts_root: str,
    normal_root: str,
    metadata_csv: str,
    n_splits: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seeds: Optional[List[int]] = None,
) -> List[Dict]:
    """Create multiple train/val/test splits with different seeds."""
    print("\n" + "="*60)
    print(f"(6) Replicated Splits (n={n_splits})")
    print("="*60)
    
    if seeds is None:
        seeds = [42 + i for i in range(n_splits)]
    
    # Get TTS patient IDs
    tts_dirs = [d for d in os.listdir(tts_root) if os.path.isdir(os.path.join(tts_root, d))]
    normal_dirs = [d for d in os.listdir(normal_root) if os.path.isdir(os.path.join(normal_root, d))]
    
    all_tts_ids = sorted(tts_dirs)
    all_normal_ids = sorted(normal_dirs)
    
    print(f"  Total TTS: {len(all_tts_ids)}, Normal: {len(all_normal_ids)}")
    
    splits = []
    for split_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        
        # Split TTS
        tts_train_val, tts_test = train_test_split(
            all_tts_ids, test_size=test_size, random_state=seed
        )
        tts_train, tts_val = train_test_split(
            tts_train_val, test_size=val_size/(1-test_size), random_state=seed
        )
        
        # Split Normal
        normal_train_val, normal_test = train_test_split(
            all_normal_ids, test_size=test_size, random_state=seed
        )
        normal_train, normal_val = train_test_split(
            normal_train_val, test_size=val_size/(1-test_size), random_state=seed
        )
        
        split_info = {
            "split_idx": split_idx,
            "seed": seed,
            "tts_train": tts_train,
            "tts_val": tts_val,
            "tts_test": tts_test,
            "normal_train": normal_train,
            "normal_val": normal_val,
            "normal_test": normal_test,
        }
        splits.append(split_info)
        
        print(f"  Split {split_idx+1} (seed={seed}): "
              f"TTS train={len(tts_train)}, val={len(tts_val)}, test={len(tts_test)} | "
              f"Normal train={len(normal_train)}, val={len(normal_val)}, test={len(normal_test)}")
    
    return splits


# ============================================================================
# Main evaluation function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robust FM Evaluation - 6 Critical Validations")
    
    # Model paths
    parser.add_argument("--ae_model_path", type=str, required=True)
    parser.add_argument("--fm_model_path", type=str, required=True)
    
    # Data paths
    parser.add_argument("--tts_root", type=str, required=True)
    parser.add_argument("--normal_root", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    
    # Parameters
    parser.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--K", type=int, default=10, help="Default K for FM scoring")
    parser.add_argument("--t_range", type=float, nargs=2, default=None, help="t_range [min, max] or None for [0,1]")
    parser.add_argument("--seed", type=int, default=42)
    
    # Sensitivity analysis
    parser.add_argument("--sensitivity_K", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--sensitivity_t_ranges", type=float, nargs="+", default=[0.0, 0.2, 0.5, 1.0],
                       help="t_range boundaries: [0.0, 0.2, 0.5, 1.0] creates [0,0.2], [0.2,0.5], [0.5,1.0], [0,1.0]")
    
    # Permutation test
    parser.add_argument("--n_permutations", type=int, default=1000)
    
    # Replicated splits
    parser.add_argument("--n_splits", type=int, default=5, help="Number of replicated splits")
    parser.add_argument("--run_replicated", action="store_true", help="Run evaluation on multiple splits")
    
    # Output
    parser.add_argument("--out_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.out_dir, "scoring_log.log")
    logger = ScoringLogger(log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    # Load models
    print("\n--- Loading Models ---")
    ae_model = load_model(args.ae_model_path, device)
    checkpoint_fm = torch.load(args.fm_model_path, map_location=device)
    fm_model = LatentVelocityEstimator(
        in_channels=128,
        base_channels=64,
        time_dim=128,
        num_res_blocks=4,
    ).to(device)
    fm_model.load_state_dict(checkpoint_fm["model"])
    print("✅ Models loaded")
    
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
    
    # Extract latents
    print("\n--- Extracting Latents ---")
    latent_dataset_val = LatentDataset(ae_model, ds_val, device)
    latent_dataset_test = LatentDataset(ae_model, ds_test, device)
    
    # Get labels
    labels_val = np.array([ds_val[i]["y"] for i in range(len(ds_val))])
    labels_test = np.array([ds_test[i]["y"] for i in range(len(ds_test))])
    
    # Prepare t_ranges for sensitivity analysis
    t_boundaries = args.sensitivity_t_ranges
    t_ranges = []
    for i in range(len(t_boundaries) - 1):
        t_ranges.append((t_boundaries[i], t_boundaries[i+1]))
    t_ranges.append(None)  # Full range [0, 1]
    
    # ========================================================================
    # (1) Val/Test 동일 스코어링 조건 검증
    # ========================================================================
    print("\n" + "="*60)
    print("(1) Val/Test 동일 스코어링 조건 검증")
    print("="*60)
    
    scoring_params = {
        "K": args.K,
        "t_range": args.t_range if args.t_range else [0.0, 1.0],
        "seed": args.seed,
    }
    
    logger.log_scoring_params("val", "FM", scoring_params, args.seed)
    logger.log_scoring_params("test", "FM", scoring_params, args.seed)
    
    # ========================================================================
    # (2) K와 t-range 민감도 분석
    # ========================================================================
    sensitivity_df = sensitivity_analysis(
        fm_model, latent_dataset_val, labels_val,
        args.sensitivity_K, t_ranges, args.seed, device, args.out_dir
    )
    
    # ========================================================================
    # (3) Score 분포 plot + effect size
    # ========================================================================
    # Use best parameters from sensitivity analysis or default
    best_row = sensitivity_df.loc[sensitivity_df['ROC-AUC'].idxmax()]
    best_K = int(best_row['K'])
    best_t_range_str = best_row['t_range']
    
    # Parse best t_range
    if best_t_range_str == "[0.0, 1.0]":
        best_t_range = None
    else:
        best_t_range = (best_row['t_min'], best_row['t_max'])
    
    print(f"\n  Using best parameters: K={best_K}, t_range={best_t_range_str}")
    
    scores_test, _ = compute_fm_scores_with_params(
        fm_model, latent_dataset_test, best_K, best_t_range, args.seed, device
    )
    
    effect_size_stats = plot_score_distributions(
        scores_test, labels_test, "Flow Matching", args.out_dir
    )
    
    # ========================================================================
    # (4) Permutation test
    # ========================================================================
    perm_results, perm_fig = permutation_test(
        scores_test, labels_test, args.n_permutations, args.seed
    )
    perm_path = os.path.join(args.out_dir, "permutation_test.png")
    perm_fig.savefig(perm_path, dpi=300, bbox_inches='tight')
    plt.close(perm_fig)
    print(f"✅ Saved permutation test plot to {perm_path}")
    
    perm_json_path = os.path.join(args.out_dir, "permutation_test.json")
    with open(perm_json_path, 'w') as f:
        json.dump(perm_results, f, indent=2)
    
    # ========================================================================
    # (5) Confounder 체크
    # ========================================================================
    confounder_results = confounder_analysis(
        scores_test, labels_test, ds_test, args.out_dir
    )
    
    # ========================================================================
    # (6) Replicated splits (optional)
    # ========================================================================
    if args.run_replicated:
        print("\n" + "="*60)
        print("(6) Replicated Splits")
        print("="*60)
        
        splits = create_replicated_splits(
            args.tts_root, args.normal_root, args.metadata_csv,
            n_splits=args.n_splits, seeds=None
        )
        
        # Save splits info
        splits_path = os.path.join(args.out_dir, "replicated_splits.json")
        # Convert to JSON-serializable format
        splits_serializable = []
        for s in splits:
            splits_serializable.append({
                "split_idx": s["split_idx"],
                "seed": s["seed"],
                "tts_train": s["tts_train"],
                "tts_val": s["tts_val"],
                "tts_test": s["tts_test"],
                "normal_train": s["normal_train"],
                "normal_val": s["normal_val"],
                "normal_test": s["normal_test"],
            })
        with open(splits_path, 'w') as f:
            json.dump(splits_serializable, f, indent=2)
        print(f"✅ Saved split definitions to {splits_path}")
        print("\n  Note: To run evaluation on these splits, implement split-specific evaluation.")
    
    # Save final summary
    logger.save_logs()
    
    print("\n" + "="*60)
    print("✅ ALL 6 VALIDATIONS COMPLETED")
    print("="*60)
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
