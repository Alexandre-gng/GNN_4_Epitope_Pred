"""
Cross-Validation for GCN model.

This module provides a function to perform k-fold cross-validation on the GCN model,
saving each fold's model state and computing Out-Of-Fold (OOF) metrics.

Adapted from MGAT_CV.py — simplified for a single-view GCN (no multi-view dataset).
"""

import os
import json
from dataclasses import asdict
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
)

from GCN import PyGGCNModel
from GCN_func import (
    TrainingConfig,
    train_n_epochs,
    evaluate,
    evaluate_w_threshold,
    find_best_threshold,
    compute_predictions,
)


def cross_validate_gcn(
    train_data: List,
    val_data: List,
    blind_test_data: List,
    model_name: str,
    config: TrainingConfig,
    device: torch.device,
    project_root: str,
    use_focal_loss: bool = False,
    use_edge_weights: bool = False, 
    use_mask: bool = True,
    in_channels: int = 1792,
) -> Dict:
    """
    Perform k-fold cross-validation on GCN model and save OOF predictions and metrics.

    Data split strategy:
        - CV is performed on combined train + val datasets.
        - Each fold splits these samples into train/val subsets.
        - Blind test set (separate antigens) is used for final evaluation of each fold.
        - OOF predictions are computed on the CV validation folds only.

    Args:
        train_data: List of PyG Data objects for training.
        val_data: List of PyG Data objects for validation.
        blind_test_data: List of PyG Data objects for blind testing.
        model_name: Name of the model for saving checkpoints.
        config: Training configuration with hyperparameters.
        device: Device to run training on (CPU or GPU).
        project_root: Root directory of the project.
        use_focal_loss: Whether to use focal loss instead of BCEWithLogitsLoss.
        use_edge_weights: Whether to use edge weights.
        use_mask: Whether to use mask to filter residues.
        in_channels: Number of input features (embedding dimension).
    Returns:
        Dictionary containing OOF metrics and fold-wise results.
    """
    # Import FocalLoss from shared location
    import sys
    sys.path.insert(0, os.path.join(project_root, "src", "models", "MGAT"))
    from FOCAL_LOSS import FocalLoss

    # ------------------------------------------------------------------
    # STEP 1: Combine train and val datasets for cross-validation
    # ------------------------------------------------------------------
    print(f"train_data samples: {len(train_data)}")
    print(f"val_data samples: {len(val_data)}")
    print(f"blind_test_data samples: {len(blind_test_data)}")
    combined_data = train_data + val_data
    num_samples = len(combined_data)

    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION SETUP — {model_name}")
    print(f"{'='*80}")
    print(f"Total samples for CV (train + val): {num_samples}")
    print(f"Number of folds: {config.num_folds}")
    print(f"Samples per fold (approx): {num_samples // config.num_folds}")
    print(f"Blind test set: {len(blind_test_data)} separate antigens")
    print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # STEP 2: Initialize KFold
    # ------------------------------------------------------------------
    kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=42)

    # Storage for OOF predictions (residue-level)
    all_oof_preds = []
    all_oof_labels = []

    # Storage for fold results
    fold_results = []

    # Create output directory
    model_dir = os.path.join(project_root, "src", "models", "trained_model", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 3: K-Fold loop
    # ------------------------------------------------------------------
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(num_samples))):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1}/{config.num_folds}")
        print(f"{'='*80}")
        print(f"  Training samples:          {len(train_indices)}")
        print(f"  Validation samples (OOF):  {len(val_indices)}")
        print(f"  Blind test samples:        {len(blind_test_data)}")

        # Split combined dataset for this fold
        train_fold = [combined_data[i] for i in train_indices]
        val_fold = [combined_data[i] for i in val_indices]

        # Create data loaders
        train_loader = DataLoader(train_fold, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=config.batch_size, shuffle=False)
        blind_test_loader = DataLoader(blind_test_data, batch_size=config.batch_size, shuffle=False)

        # Initialize model
        model = PyGGCNModel(
            n_features=in_channels,
            n_edge_attr=config.n_edge_attr,
            n_hidden=config.n_hidden,
            n_layers=config.n_layers,
            dropout=config.dropout,
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Criterion
        if use_focal_loss:
            criterion = FocalLoss(alpha=config.alpha, gamma=config.gamma, reduction="mean")
        else:
            criterion = nn.BCEWithLogitsLoss()

        # ---- Train ----
        best_model_state = train_n_epochs(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            blind_test_loader=blind_test_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            use_mask=use_mask,
        )

        # Load best model state
        model.load_state_dict(best_model_state)

        # ---- Verification ----
        print(f"\n>>> Verifying loaded best model for fold {fold_idx + 1}...")
        verify_val_loss, verify_val_ap, verify_val_roc = evaluate(
            model, val_loader, criterion, device, use_mask=use_mask,
        )
        verify_test_loss, verify_test_ap, verify_test_roc = evaluate(
            model, blind_test_loader, criterion, device, use_mask=use_mask,
        )
        print(f">>> Verification — Val AP: {verify_val_ap:.4f}, Val ROC: {verify_val_roc:.4f}")
        print(f">>> Verification — Blind Test AP: {verify_test_ap:.4f}, Blind Test ROC: {verify_test_roc:.4f}")

        # ---- Best threshold (MCC) on val set ----
        best_threshold = find_best_threshold(
            model, val_loader, criterion, device, use_mask=use_mask,
        )
        print(f"Best threshold for fold {fold_idx + 1}: {best_threshold:.4f}")

        # ---- OOF predictions ----
        oof_preds, oof_labels, oof_loss = compute_predictions(
            model, val_loader, criterion, device, use_mask=use_mask,
        )
        all_oof_preds.append(oof_preds)
        all_oof_labels.append(oof_labels)

        # ---- Evaluate with threshold ----
        val_loss, val_auc_pr, val_auc_roc, val_mcc, val_f1, val_acc = evaluate_w_threshold(
            model, val_loader, criterion, device, best_threshold, use_mask=use_mask,
        )
        test_loss, test_auc_pr, test_auc_roc, test_mcc, test_f1, test_acc = evaluate_w_threshold(
            model, blind_test_loader, criterion, device, best_threshold, use_mask=use_mask,
        )

        # Sanity check
        val_ap_diff = abs(val_auc_pr - verify_val_ap)
        test_ap_diff = abs(test_auc_pr - verify_test_ap)
        if val_ap_diff > 0.001 or test_ap_diff > 0.001:
            print(f"\n  WARNING: Metrics mismatch detected!")
            print(f"   Val AP diff: {val_ap_diff:.6f}  |  Test AP diff: {test_ap_diff:.6f}")
        else:
            print(f"  Metrics verification passed (differences < 0.001)")

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Validation  — Loss: {val_loss:.4f}, AP: {val_auc_pr:.4f}, ROC-AUC: {val_auc_roc:.4f}, MCC: {val_mcc:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        print(f"  Blind Test  — Loss: {test_loss:.4f}, AP: {test_auc_pr:.4f}, ROC-AUC: {test_auc_roc:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}")

        # Save fold results
        fold_result = {
            "fold": fold_idx + 1,
            "threshold": float(best_threshold),
            "validation": {
                "loss": float(val_loss),
                "auc_pr": float(val_auc_pr),
                "auc_roc": float(val_auc_roc),
                "mcc": float(val_mcc),
                "f1": float(val_f1),
                "accuracy": float(val_acc),
            },
            "blind_test": {
                "loss": float(test_loss),
                "auc_pr": float(test_auc_pr),
                "auc_roc": float(test_auc_roc),
                "mcc": float(test_mcc),
                "f1": float(test_f1),
                "accuracy": float(test_acc),
            },
        }
        fold_results.append(fold_result)

        # Running uncertainty info for AUC-PR across completed folds (SEM)
        running_val_auc_pr = [f["validation"]["auc_pr"] for f in fold_results]
        running_test_auc_pr = [f["blind_test"]["auc_pr"] for f in fold_results]
        if len(running_val_auc_pr) > 1:
            running_val_auc_pr_sem = float(np.std(running_val_auc_pr, ddof=1) / np.sqrt(len(running_val_auc_pr)))
            running_test_auc_pr_sem = float(np.std(running_test_auc_pr, ddof=1) / np.sqrt(len(running_test_auc_pr)))
            print(
                f"  Running AUC-PR SEM after fold {fold_idx + 1}: "
                f"Val={running_val_auc_pr_sem:.4f}, Blind Test={running_test_auc_pr_sem:.4f}"
            )
        else:
            print(f"  Running AUC-PR SEM after fold {fold_idx + 1}: Val=N/A, Blind Test=N/A")

        # Save fold checkpoint
        checkpoint = {
            "fold": fold_idx + 1,
            "model_state_dict": best_model_state,
            "config": asdict(config),
            "threshold": float(best_threshold),
            "metrics": fold_result,
        }
        fold_path = os.path.join(model_dir, f"fold_{fold_idx + 1}.pt")
        torch.save(checkpoint, fold_path)
        print(f"Saved fold {fold_idx + 1} checkpoint to {fold_path}")

    # ------------------------------------------------------------------
    # STEP 4: Compute OOF metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("STEP 4: Computing Out-Of-Fold (OOF) Metrics")
    print(f"{'='*80}")

    if len(all_oof_preds) == 0 or len(all_oof_labels) == 0:
        raise ValueError("No OOF predictions collected.")

    all_oof_preds = np.concatenate(all_oof_preds)
    all_oof_labels = np.concatenate(all_oof_labels)

    print(f"Total OOF residues: {len(all_oof_preds)}")
    print(f"Positive labels: {int(np.sum(all_oof_labels))} ({100*np.mean(all_oof_labels):.2f}%)")
    print(f"Negative labels: {len(all_oof_labels) - int(np.sum(all_oof_labels))} ({100*(1-np.mean(all_oof_labels)):.2f}%)")

    if len(np.unique(all_oof_labels)) < 2:
        raise ValueError(f"OOF labels contain only one class: {np.unique(all_oof_labels)}")

    oof_auc_pr = average_precision_score(all_oof_labels, all_oof_preds)
    oof_auc_roc = roc_auc_score(all_oof_labels, all_oof_preds)

    # Find best OOF threshold (maximise MCC)
    best_oof_threshold = 0.5
    best_oof_mcc = -1.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        binary_preds = (all_oof_preds >= threshold).astype(int)
        try:
            mcc_score = matthews_corrcoef(all_oof_labels, binary_preds)
            if mcc_score > best_oof_mcc:
                best_oof_mcc = mcc_score
                best_oof_threshold = threshold
        except Exception:
            continue

    print(f"\nBest OOF threshold: {best_oof_threshold:.4f} (MCC: {best_oof_mcc:.4f})")

    oof_binary_preds = (all_oof_preds >= best_oof_threshold).astype(int)
    oof_mcc = matthews_corrcoef(all_oof_labels, oof_binary_preds)
    oof_f1 = f1_score(all_oof_labels, oof_binary_preds)
    oof_acc = accuracy_score(all_oof_labels, oof_binary_preds)

    print(f"\nOOF Metrics (threshold={best_oof_threshold:.4f}):")
    print(f"  AUC-PR:   {oof_auc_pr:.4f}")
    print(f"  AUC-ROC:  {oof_auc_roc:.4f}")
    print(f"  MCC:      {oof_mcc:.4f}")
    print(f"  F1:       {oof_f1:.4f}")
    print(f"  Accuracy: {oof_acc:.4f}")

    if len(fold_results) == 0:
        raise ValueError("No fold results collected.")

    # ------------------------------------------------------------------
    # Aggregate fold metrics
    # ------------------------------------------------------------------
    def _mean(key_path):
        return np.mean([f[key_path[0]][key_path[1]] for f in fold_results])

    avg_val_metrics = {
        k: float(_mean(("validation", k)))
        for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")
    }
    avg_test_metrics = {
        k: float(_mean(("blind_test", k)))
        for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")
    }

    def _sem(key_path):
        values = [f[key_path[0]][key_path[1]] for f in fold_results]
        if len(values) < 2:
            return 0.0
        return np.std(values, ddof=1) / np.sqrt(len(values))

    sem_val_metrics = {
        k: float(_sem(("validation", k)))
        for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")
    }
    sem_test_metrics = {
        k: float(_sem(("blind_test", k)))
        for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")
    }

    print(f"\nAverage Validation Metrics across folds:")
    for k, v in avg_val_metrics.items():
        print(f"  {k:>10s}: {v:.4f}")

    print(f"\nValidation Metrics SEM across folds:")
    for k, v in sem_val_metrics.items():
        print(f"  {k:>10s}: {v:.4f}")

    print(f"\nAverage Blind Test Metrics across folds:")
    for k, v in avg_test_metrics.items():
        print(f"  {k:>10s}: {v:.4f}")

    print(f"\nBlind Test Metrics SEM across folds:")
    for k, v in sem_test_metrics.items():
        print(f"  {k:>10s}: {v:.4f}")

    # ------------------------------------------------------------------
    # Save OOF report
    # ------------------------------------------------------------------
    oof_metrics = {
        "model_name": model_name,
        "num_folds": config.num_folds,
        "oof_threshold": float(best_oof_threshold),
        "oof_metrics": {
            "auc_pr": float(oof_auc_pr),
            "auc_roc": float(oof_auc_roc),
            "mcc": float(oof_mcc),
            "f1": float(oof_f1),
            "accuracy": float(oof_acc),
        },
        "average_validation_metrics": avg_val_metrics,
        "sem_validation_metrics": sem_val_metrics,
        "average_blind_test_metrics": avg_test_metrics,
        "sem_blind_test_metrics": sem_test_metrics,
        "fold_results": fold_results,
        "config": asdict(config),
    }

    oof_metrics_path = os.path.join(model_dir, "OOF_metrics.json")
    with open(oof_metrics_path, "w") as f:
        json.dump(oof_metrics, f, indent=4)

    print(f"\nSaved OOF metrics to {oof_metrics_path}")

    return oof_metrics
