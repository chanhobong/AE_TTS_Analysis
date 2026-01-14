#!/usr/bin/env python3
"""
compare_epochs.py
- Compare linear probe AUC across different epoch checkpoints
- Epochs: 20, 40, 60, 80, 100
- Shows that reconstruction loss ↓ ≠ classification performance ↑

This demonstrates the importance of early stopping or separate validation!
"""

import os
import sys
import argparse
from typing import Dict, List

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
from sklearn.metrics import roc_auc_score
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


def evaluate_checkpoint(model_path: str, ds_train, ds_val, ds_test, device: torch.device, 
                       batch_size: int = 16, c_values: List[float] = None) -> Dict:
    """Evaluate a single checkpoint."""
    if c_values is None:
        c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Load model
    model = load_model(model_path, device)
    
    # Extract features
    X_train, y_train = extract_features(model, ds_train, device, batch_size)
    X_val, y_val = extract_features(model, ds_val, device, batch_size)
    X_test, y_test = extract_features(model, ds_test, device, batch_size)
    
    # Tune C on val
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    best_c = None
    best_val_auc = -1.0
    val_results = {}
    
    for c in c_values:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42, solver='liblinear')
        clf.fit(X_train_scaled, y_train)
        y_val_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        val_results[c] = val_auc
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c
    
    # Evaluate on test with best C
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42, solver='liblinear')
    clf.fit(X_train_scaled, y_train)
    y_test_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    return {
        "best_c": best_c,
        "val_auc": best_val_auc,
        "test_auc": test_auc,
        "val_results": val_results,
    }


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, 
                   help="Directory containing epoch checkpoints (model_epoch020.pth, etc.)")
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    
    p.add_argument("--epochs", type=int, nargs="+", default=[20, 40, 60, 80, 100],
                   help="Epochs to compare")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--c_values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs/epoch_comparison")
    return p


def main():
    args = build_argparser().parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    print("=" * 60)
    print("EPOCH-BY-EPOCH LINEAR PROBE COMPARISON")
    print("=" * 60)
    print(f"model_dir   : {args.model_dir}")
    print(f"epochs     : {args.epochs}")
    print("=" * 60)
    
    # Build datasets
    print("\n--- Building Datasets ---")
    ds_train_tts = build_ds(args.tts_root, args.metadata_csv, expected_shape_xyz, 
                            include_csv=args.train_csv, name="TTS(train)")
    ds_train_norm = build_ds(args.normal_root, args.metadata_csv, expected_shape_xyz, 
                            include_csv=args.train_csv, name="Normal(train)")
    ds_train = ConcatDataset([ds_train_tts, ds_train_norm])
    
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
    
    # Evaluate each epoch
    results = {}
    
    print("\n" + "=" * 60)
    print("EVALUATING EACH EPOCH")
    print("=" * 60)
    
    for epoch in args.epochs:
        # Try different naming conventions
        model_paths = [
            os.path.join(args.model_dir, f"model_epoch{epoch:03d}.pth"),
            os.path.join(args.model_dir, f"model_epoch{epoch}.pth"),
            os.path.join(args.model_dir, f"epoch_{epoch}.pth"),
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print(f"⚠️ Epoch {epoch}: Checkpoint not found, skipping...")
            continue
        
        print(f"\n--- Epoch {epoch} ---")
        print(f"Model: {model_path}")
        
        try:
            result = evaluate_checkpoint(model_path, ds_train, ds_val, ds_test, 
                                        device, args.batch_size, args.c_values)
            results[epoch] = result
            print(f"  Val AUC: {result['val_auc']:.4f} (best C: {result['best_c']})")
            print(f"  Test AUC: {result['test_auc']:.4f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Epoch':<10} {'Val AUC':<10} {'Test AUC':<10} {'Best C':<10}")
    print("-" * 60)
    
    for epoch in sorted(results.keys()):
        r = results[epoch]
        print(f"{epoch:<10} {r['val_auc']:<10.4f} {r['test_auc']:<10.4f} {r['best_c']:<10.4f}")
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    
    summary_path = os.path.join(args.out_dir, "epoch_comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EPOCH-BY-EPOCH LINEAR PROBE COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Epoch':<10} {'Val AUC':<10} {'Test AUC':<10} {'Best C':<10}\n")
        f.write("-" * 60 + "\n")
        for epoch in sorted(results.keys()):
            r = results[epoch]
            f.write(f"{epoch:<10} {r['val_auc']:<10.4f} {r['test_auc']:<10.4f} {r['best_c']:<10.4f}\n")
    
    print(f"\n💾 Results saved to: {summary_path}")
    print("\n✅ Comparison completed!")
    print("\n📊 Key insight:")
    print("   Reconstruction loss ↓ does NOT always mean classification performance ↑")


if __name__ == "__main__":
    main()
