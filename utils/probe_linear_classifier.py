#!/usr/bin/env python3
"""
probe_linear_classifier.py
- Load trained AE model and extract features using encoder
- Train linear classifier (Logistic Regression) on train features
- Tune C hyperparameter on val set
- Evaluate final performance on test set (1 time only)

Example:
  python probe_linear_classifier.py \
    --model_path /path/to/model.pth \
    --tts_root /path/to/TTS_RM_V2 \
    --normal_root /path/to/normal_RM_V2 \
    --metadata_csv /path/to/Combined_Labels.csv \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --test_csv /path/to/test.csv
"""

import os
import sys
import random
import argparse
from typing import List, Tuple, Dict

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

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

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(base, base * 2, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),

            nn.Conv3d(base * 2, base * 4, kernel_size=4, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),

            nn.Conv3d(base * 4, base * 8, kernel_size=4, stride=2, padding=1),  # /8
            nn.ReLU(inplace=True),
        )

        # Decoder (not used for feature extraction, but needed for loading)
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
        """Extract features from encoder."""
        z = self.enc(x)
        # Global Average Pooling to get fixed-size feature vector
        # z shape: (B, base*8, Z/8, Y/8, X/8)
        z_pooled = nn.functional.adaptive_avg_pool3d(z, (1, 1, 1))  # (B, base*8, 1, 1, 1)
        z_flat = z_pooled.view(z_pooled.size(0), -1)  # (B, base*8)
        return z_flat


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load trained AE model."""
    print(f"📦 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model args if available
    if "args" in checkpoint:
        args = checkpoint["args"]
        base = args.get("base", 16) if isinstance(args, dict) else 16
    else:
        base = 16  # default
    
    model = Tiny3DAE(in_ch=1, base=base).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"✅ Model loaded successfully (base={base})")
    return model


@torch.no_grad()
def extract_features(model: nn.Module, dataset, device: torch.device, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from dataset.
    
    Returns:
        features: (N, feature_dim) numpy array
        labels: (N,) numpy array
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    
    all_features = []
    all_labels = []
    
    print(f"  Extracting features from {len(dataset)} samples...")
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)  # (B, 1, Z, Y, X)
        y = batch["y"].numpy()  # (B,)
        
        # Extract features
        features = model.encode(x)  # (B, feature_dim)
        features_np = features.cpu().numpy()
        
        all_features.append(features_np)
        all_labels.append(y)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Processed {batch_idx + 1} batches...")
    
    features = np.concatenate(all_features, axis=0)  # (N, feature_dim)
    labels = np.concatenate(all_labels, axis=0)  # (N,)
    
    print(f"  ✅ Extracted features shape: {features.shape}, labels shape: {labels.shape}")
    return features, labels


def build_ds(root_dir, metadata_csv, expected_xyz, include_csv="", name=""):
    """Build dataset with optional include_patient_csv filter."""
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


def tune_c_hyperparameter(X_train, y_train, X_val, y_val, c_values: List[float]) -> Tuple[float, Dict]:
    """
    Tune C hyperparameter on validation set.
    
    Returns:
        best_c: Best C value
        results: Dictionary with results for each C
    """
    print("\n" + "=" * 60)
    print("TUNING C HYPERPARAMETER ON VAL SET")
    print("=" * 60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    results = {}
    best_c = None
    best_auc = -1.0
    
    for c in c_values:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42, solver='liblinear')
        clf.fit(X_train_scaled, y_train)
        
        # Predict on val set
        y_val_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        
        y_val_pred = clf.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, zero_division=0)
        val_rec = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        results[c] = {
            "auc": val_auc,
            "acc": val_acc,
            "prec": val_prec,
            "rec": val_rec,
            "f1": val_f1,
        }
        
        print(f"C={c:8.6f} | Val AUC={val_auc:.4f} | Acc={val_acc:.4f} | Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_c = c
    
    print(f"\n✅ Best C: {best_c} (Val AUC: {best_auc:.4f})")
    return best_c, results


def evaluate_on_test(X_train, y_train, X_test, y_test, best_c: float, scaler: StandardScaler = None) -> Dict:
    """
    Train final model with best C and evaluate on test set (1 time only).
    
    Returns:
        Dictionary with test metrics
    """
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET (1 TIME ONLY)")
    print("=" * 60)
    
    # Standardize features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train final model
    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42, solver='liblinear')
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate on test
    y_test_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = clf.predict(X_test_scaled)
    
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    
    results = {
        "auc": test_auc,
        "acc": test_acc,
        "prec": test_prec,
        "rec": test_rec,
        "f1": test_f1,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
    
    print(f"Test AUC:  {test_auc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Test Prec: {test_prec:.4f}")
    print(f"Test Rec:  {test_rec:.4f}")
    print(f"Test F1:   {test_f1:.4f}")
    
    return results


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to trained model.pth")
    p.add_argument("--tts_root", type=str, required=True, help="Root directory for TTS dataset")
    p.add_argument("--normal_root", type=str, required=True, help="Root directory for Normal dataset")
    p.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file")
    p.add_argument("--train_csv", type=str, required=True, help="Path to train.csv split file")
    p.add_argument("--val_csv", type=str, required=True, help="Path to val.csv split file")
    p.add_argument("--test_csv", type=str, required=True, help="Path to test.csv split file")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64], metavar=("X", "Y", "Z"))
    
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    p.add_argument("--c_values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 
                   help="C values to try for hyperparameter tuning")
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--out_dir", type=str, default="./outputs/probe_results", help="Output directory for results")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device: {device}")
    
    expected_shape_xyz = tuple(args.expected_shape)
    
    # Check files exist
    for name, path in [("model", args.model_path), ("train_csv", args.train_csv), 
                       ("val_csv", args.val_csv), ("test_csv", args.test_csv)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    
    print("=" * 60)
    print("LINEAR PROBE CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"model_path   : {args.model_path}")
    print(f"tts_root     : {args.tts_root}")
    print(f"normal_root  : {args.normal_root}")
    print(f"metadata_csv : {args.metadata_csv}")
    print(f"train_csv    : {args.train_csv}")
    print(f"val_csv      : {args.val_csv}")
    print(f"test_csv     : {args.test_csv}")
    print(f"expected_xyz : {expected_shape_xyz}")
    print("=" * 60)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Build datasets
    print("\n" + "=" * 60)
    print("BUILDING DATASETS")
    print("=" * 60)
    
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
    
    print(f"\n✅ Train size: {len(ds_train)}")
    print(f"✅ Val size: {len(ds_val)}")
    print(f"✅ Test size: {len(ds_test)}")
    
    # Extract features
    print("\n" + "=" * 60)
    print("EXTRACTING FEATURES")
    print("=" * 60)
    
    print("\n--- Train set ---")
    X_train, y_train = extract_features(model, ds_train, device, args.batch_size)
    
    print("\n--- Val set ---")
    X_val, y_val = extract_features(model, ds_val, device, args.batch_size)
    
    print("\n--- Test set ---")
    X_test, y_test = extract_features(model, ds_test, device, args.batch_size)
    
    # Tune C on val set
    best_c, val_results = tune_c_hyperparameter(X_train, y_train, X_val, y_val, args.c_values)
    
    # Final evaluation on test set (1 time only)
    scaler = StandardScaler()
    scaler.fit(X_train)
    test_results = evaluate_on_test(X_train, y_train, X_test, y_test, best_c, scaler)
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save summary
    summary_path = os.path.join(args.out_dir, "probe_results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LINEAR PROBE CLASSIFIER RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Best C (tuned on val): {best_c}\n\n")
        
        f.write("Validation Set Results (C tuning):\n")
        f.write("-" * 60 + "\n")
        for c, metrics in sorted(val_results.items()):
            f.write(f"C={c:8.6f} | AUC={metrics['auc']:.4f} | Acc={metrics['acc']:.4f} | "
                   f"Prec={metrics['prec']:.4f} | Rec={metrics['rec']:.4f} | F1={metrics['f1']:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Test Set Results (Final, 1 time only):\n")
        f.write("-" * 60 + "\n")
        f.write(f"AUC:  {test_results['auc']:.4f}\n")
        f.write(f"Acc:  {test_results['acc']:.4f}\n")
        f.write(f"Prec: {test_results['prec']:.4f}\n")
        f.write(f"Rec:  {test_results['rec']:.4f}\n")
        f.write(f"F1:   {test_results['f1']:.4f}\n")
    
    print(f"\n💾 Results saved to: {summary_path}")
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
