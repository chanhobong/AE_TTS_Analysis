#!/usr/bin/env python3
"""
compare_normal_vs_mixed.py
- Compare Normal-only AE vs Mixed (TTS+Normal) AE
- Classic anomaly detection baseline: "Normal만으로 학습하면 TTS가 더 잘 튀어나오지 않을까?"

This is a very important experiment for the paper!
"""

import os
import sys
import argparse
from typing import Dict, Tuple

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from data.dataset import CTROIVolumeDataset

# Import from probe_linear_classifier (avoid circular import)
import importlib.util
spec = importlib.util.spec_from_file_location("probe_linear_classifier", 
    os.path.join(os.path.dirname(__file__), "probe_linear_classifier.py"))
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)

Tiny3DAE = probe_module.Tiny3DAE
load_model = probe_module.load_model
extract_features = probe_module.extract_features
build_ds = probe_module.build_ds


def evaluate_model_with_probe(model_path: str, ds_train, ds_val, ds_test, device: torch.device,
                               batch_size: int = 16, c_values: list = None) -> Dict:
    """Evaluate a model with linear probe (requires both classes in train set)."""
    if c_values is None:
        c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Load model
    model = load_model(model_path, device)
    
    # Extract features
    X_train, y_train = extract_features(model, ds_train, device, batch_size)
    X_val, y_val = extract_features(model, ds_val, device, batch_size)
    X_test, y_test = extract_features(model, ds_test, device, batch_size)
    
    # Check if train set has both classes
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        raise ValueError(f"Train set has only {len(unique_labels)} class(es). Linear probe requires both classes.")
    
    # Tune C on val
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    best_c = None
    best_val_auc = -1.0
    
    for c in c_values:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42, solver='liblinear')
        clf.fit(X_train_scaled, y_train)
        y_val_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c
    
    # Evaluate on test
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42, solver='liblinear')
    clf.fit(X_train_scaled, y_train)
    y_test_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = clf.predict(X_test_scaled)
    
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    return {
        "best_c": best_c,
        "val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "test_f1": test_f1,
    }


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


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_normal", type=str, required=True,
                   help="Path to Normal-only trained model")
    p.add_argument("--model_mixed", type=str, required=True,
                   help="Path to Mixed (TTS+Normal) trained model")
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--c_values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs/normal_vs_mixed")
    return p


def main():
    args = build_argparser().parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("NORMAL-ONLY vs MIXED AE COMPARISON")
    print("=" * 60)
    print(f"Normal model: {args.model_normal}")
    print(f"Mixed model:  {args.model_mixed}")
    print("=" * 60)
    
    # Build datasets
    print("\n--- Building Datasets ---")
    
    # For Normal-only: train on Normal only
    ds_train_normal_only = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                                    include_csv=args.train_csv, name="Normal(train, for normal-only)")
    
    # For Mixed: train on TTS + Normal
    ds_train_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz,
                           include_csv=args.train_csv, name="TTS(train, for mixed)")
    ds_train_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz,
                            include_csv=args.train_csv, name="Normal(train, for mixed)")
    ds_train_mixed = ConcatDataset([ds_train_tts, ds_train_norm])
    
    # Val and test (both use full data)
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
    
    print(f"\n✅ Train sizes: Normal-only={len(ds_train_normal_only)}, Mixed={len(ds_train_mixed)}")
    print(f"✅ Val size: {len(ds_val)}")
    print(f"✅ Test size: {len(ds_test)}")
    
    # Load models for reconstruction error comparison
    model_normal = load_model(args.model_normal, device)
    model_mixed = load_model(args.model_mixed, device)
    
    # Evaluate Normal-only model (reconstruction error only, no linear probe)
    print("\n" + "=" * 60)
    print("EVALUATING NORMAL-ONLY MODEL")
    print("=" * 60)
    print("⚠️  Note: Normal-only model trained on Normal data only.")
    print("    Cannot use linear probe (requires both classes).")
    print("    Using reconstruction error for comparison instead.")
    
    # Evaluate Mixed model with linear probe
    print("\n" + "=" * 60)
    print("EVALUATING MIXED MODEL (with linear probe)")
    print("=" * 60)
    try:
        results_mixed = evaluate_model_with_probe(args.model_mixed, ds_train_mixed, ds_val, ds_test,
                                                  device, args.batch_size, args.c_values)
        print(f"Val AUC:  {results_mixed['val_auc']:.4f} (best C: {results_mixed['best_c']})")
        print(f"Test AUC: {results_mixed['test_auc']:.4f}")
        print(f"Test Acc: {results_mixed['test_acc']:.4f}")
        print(f"Test F1:  {results_mixed['test_f1']:.4f}")
    except Exception as e:
        print(f"⚠️  Error in linear probe: {e}")
        results_mixed = None
    
    # Compare reconstruction errors
    print("\n" + "=" * 60)
    print("COMPARING RECONSTRUCTION ERRORS (KEY METRIC)")
    print("=" * 60)
    print("This is the main comparison for Normal-only vs Mixed!")
    
    print("\n--- Normal-only model ---")
    errors_normal, labels_normal = compute_reconstruction_error(model_normal, ds_test, device, args.batch_size)
    tts_errors_normal = errors_normal[labels_normal == 1]
    normal_errors_normal = errors_normal[labels_normal == 0]
    print(f"  TTS mean error:    {tts_errors_normal.mean():.6f} ± {tts_errors_normal.std():.6f}")
    print(f"  Normal mean error: {normal_errors_normal.mean():.6f} ± {normal_errors_normal.std():.6f}")
    ratio_normal = tts_errors_normal.mean() / normal_errors_normal.mean()
    print(f"  Ratio (TTS/Normal): {ratio_normal:.3f}")
    print(f"  → Higher ratio = TTS stands out more (better for anomaly detection)")
    
    print("\n--- Mixed model ---")
    errors_mixed, labels_mixed = compute_reconstruction_error(model_mixed, ds_test, device, args.batch_size)
    tts_errors_mixed = errors_mixed[labels_mixed == 1]
    normal_errors_mixed = errors_mixed[labels_mixed == 0]
    print(f"  TTS mean error:    {tts_errors_mixed.mean():.6f} ± {tts_errors_mixed.std():.6f}")
    print(f"  Normal mean error: {normal_errors_mixed.mean():.6f} ± {normal_errors_mixed.std():.6f}")
    ratio_mixed = tts_errors_mixed.mean() / normal_errors_mixed.mean()
    print(f"  Ratio (TTS/Normal): {ratio_mixed:.3f}")
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Normal-only TTS/Normal ratio: {ratio_normal:.3f}")
    print(f"Mixed TTS/Normal ratio:        {ratio_mixed:.3f}")
    if ratio_normal > ratio_mixed:
        print(f"✅ Normal-only model makes TTS stand out MORE (ratio difference: {ratio_normal - ratio_mixed:.3f})")
        print("   → Supports the hypothesis: Normal-only training is better for anomaly detection!")
    else:
        print(f"⚠️  Mixed model makes TTS stand out MORE (ratio difference: {ratio_mixed - ratio_normal:.3f})")
        print("   → Mixed training might be better for this task.")
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    
    summary_path = os.path.join(args.out_dir, "normal_vs_mixed_comparison.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("NORMAL-ONLY vs MIXED AE COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Normal-only Model:\n")
        f.write("  (Trained on Normal data only - cannot use linear probe)\n")
        f.write(f"  TTS recon error:    {tts_errors_normal.mean():.6f} ± {tts_errors_normal.std():.6f}\n")
        f.write(f"  Normal recon error: {normal_errors_normal.mean():.6f} ± {normal_errors_normal.std():.6f}\n")
        f.write(f"  Ratio (TTS/Normal): {ratio_normal:.3f}\n\n")
        
        f.write("Mixed Model:\n")
        if results_mixed:
            f.write(f"  Val AUC:  {results_mixed['val_auc']:.4f} (best C: {results_mixed['best_c']})\n")
            f.write(f"  Test AUC: {results_mixed['test_auc']:.4f}\n")
            f.write(f"  Test Acc: {results_mixed['test_acc']:.4f}\n")
            f.write(f"  Test F1:  {results_mixed['test_f1']:.4f}\n")
        else:
            f.write("  Linear probe: Not available\n")
        f.write(f"  TTS recon error:    {tts_errors_mixed.mean():.6f} ± {tts_errors_mixed.std():.6f}\n")
        f.write(f"  Normal recon error: {normal_errors_mixed.mean():.6f} ± {normal_errors_mixed.std():.6f}\n")
        f.write(f"  Ratio (TTS/Normal): {ratio_mixed:.3f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Normal-only TTS/Normal ratio: {ratio_normal:.3f}\n")
        f.write(f"Mixed TTS/Normal ratio:        {ratio_mixed:.3f}\n")
        if ratio_normal > ratio_mixed:
            f.write(f"✅ Normal-only model makes TTS stand out MORE (ratio difference: {ratio_normal - ratio_mixed:.3f})\n")
            f.write("   → Supports the hypothesis: Normal-only training is better for anomaly detection!\n")
        else:
            f.write(f"⚠️  Mixed model makes TTS stand out MORE (ratio difference: {ratio_mixed - ratio_normal:.3f})\n")
            f.write("   → Mixed training might be better for this task.\n")
        f.write("\nKEY QUESTION:\n")
        f.write("Does Normal-only training make TTS stand out more?\n")
        f.write("(Higher TTS/Normal error ratio = better anomaly detection)\n")
    
    print(f"\n💾 Results saved to: {summary_path}")
    print("\n✅ Comparison completed!")
    print("\n📊 Key insight:")
    print("   Normal-only AE is a classic anomaly detection baseline.")
    print("   If TTS/Normal error ratio is higher, it suggests better anomaly detection.")


if __name__ == "__main__":
    main()
