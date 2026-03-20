"""
Cross-Validation for MGAT model.

This module provides functions to perform k-fold cross-validation on the MGAT model,
saving each fold's model state and computing Out-Of-Fold (OOF) metrics.
"""

import os
import json
from dataclasses import asdict
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score, accuracy_score

from MGAT import MultiView_GAT
from MGAT_func import (
    TrainingConfig,
    train_n_epochs,
    evaluate,
    evaluate_w_threshold,
    find_best_threshold,
    compute_predictions
)
from multiview_dataset import MultiViewDataset


def cross_validate_mgat(
    datasets_by_view: List[List],
    model_name: str,
    config: TrainingConfig,
    device: torch.device,
    project_root: str,
    use_focal_loss: bool = False,
    use_mask: bool = True,
    in_channels: int = 1792,
    out_channels: int = 1,
    num_views: int = 4,
    edge_dim: int = 1,
) -> Dict:
    """
    Perform k-fold cross-validation on MGAT model (on 200 antigens) and save OOF predictions and metrics.
    
    Data split strategy:
        - CV is performed on combined train + val datasets (200 antigens)
        - Each fold splits these 200 antigens into train/val subsets
        - Blind test set (separate antigens) is used for final evaluation of each fold
        - OOF predictions are computed on the CV validation folds only
    
    Args:
        datasets_by_view: List of lists, where each inner list contains [train_data, val_data, blind_test_data] for one view.
                         Example: [[roh1_train, roh1_val, roh1_blind_test], [roh2_train, roh2_val, roh2_blind_test], ...]
                         - train_data: Training antigens
                         - val_data: Validation antigens
                         - blind_test_data: Blind test antigens (separate set, used only for final evaluation)
        model_name: Name of the model for saving checkpoints.
        config: Training configuration with hyperparameters.
        device: Device to run training on (CPU or GPU).
        project_root: Root directory of the project.
        use_edge_attr: Whether to use edge attributes in the model.
        use_focal_loss: Whether to use focal loss instead of BCEWithLogitsLoss.
        use_mask: Whether to use mask to filter residues.
        in_channels: Number of input channels (embedding dimension).
        out_channels: Number of output channels (1 for binary classification).
        num_views: Number of views in the multi-view dataset.
        edge_dim: Dimension of edge features.
    Returns:
        Dictionary containing OOF metrics and fold-wise results.
    """
    from FOCAL_LOSS import FocalLoss
    
    # STEP 1: Combine train and val datasets for each view (200 antigens total)
    # The cross-validation will be performed on these 200 antigens
    combined_datasets_by_view = []
    blind_test_by_view = []
    
    for view_datasets in datasets_by_view:
        train_data, val_data, blind_test_data = view_datasets
        # Combine train and val for cross-validation (200 antigens)
        combined_data = train_data + val_data
        combined_datasets_by_view.append(combined_data)
        # Keep blind test separate for final evaluation
        blind_test_by_view.append(blind_test_data)
    
    # Create combined multi-view dataset (200 antigens)
    num_samples = len(combined_datasets_by_view[0])
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION SETUP")
    print(f"{'='*80}")
    print(f"Total samples for CV (train + val): {num_samples}")
    print(f"Number of folds: {config.num_folds}")
    print(f"Samples per fold (approx): {num_samples // config.num_folds}")
    print(f"Blind test set: Separate antigens (used for final evaluation)")
    print(f"{'='*80}\n")
    
    # STEP 2: Initialize KFold on the 200 antigens
    kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    
    # Storage for OOF predictions (residue-level)
    # Since each antigen has multiple residues, we store predictions in lists
    all_oof_preds = []
    all_oof_labels = []
    
    # Storage for fold results
    fold_results = []
    
    # Create output directory
    model_dir = os.path.join(project_root, "src", "models", "trained_model", model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # STEP 3: Perform K-Fold CV on the 200 antigens
    # Each fold: 80% for training, 20% for validation (OOF)
    # Blind test set is evaluated separately for each fold
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(num_samples))):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1}/{config.num_folds}")
        print(f"{'='*80}")
        print(f"CV Fold Split - Train/Val from 200 antigens:")
        print(f"  - Training samples: {len(train_indices)}")
        print(f"  - Validation samples (OOF): {len(val_indices)}")
        print(f"  - Blind test samples (final eval): {len(blind_test_by_view[0])}")
        print(f"{'='*80}")
        
        # Create train and val datasets for this fold
        train_datasets_fold = []
        val_datasets_fold = []
        
        for view_data in combined_datasets_by_view:
            train_fold = [view_data[i] for i in train_indices]
            val_fold = [view_data[i] for i in val_indices]
            train_datasets_fold.append(train_fold)
            val_datasets_fold.append(val_fold)
        
        # Create multi-view datasets
        train_dataset = MultiViewDataset(train_datasets_fold)
        val_dataset = MultiViewDataset(val_datasets_fold)
        # Blind test is a separate dataset (not used in CV, only for final evaluation)
        blind_test_dataset = MultiViewDataset(blind_test_by_view)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        # Blind test loader (separate antigens)
        blind_test_loader = DataLoader(blind_test_dataset, batch_size=config.batch_size, shuffle=False)
        
        print(f"Datasets for Fold {fold_idx + 1}:")
        print(f"  - Train loader: {len(train_dataset)} samples")
        print(f"  - Val loader (OOF): {len(val_dataset)} samples")
        print(f"  - Blind test loader: {len(blind_test_dataset)} samples")
        
        # Initialize model
        model = MultiView_GAT(
            in_channels=in_channels,
            out_channels=out_channels,
            num_views=num_views,
            hidden_channels=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            edge_dim=edge_dim,
            reg_lambda=config.reg_lambda
        ).to(device)
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        if use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Train model
        best_model_state = train_n_epochs(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            blind_test_loader=blind_test_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            use_mask=use_mask
        )
        
        # Load best model state
        model.load_state_dict(best_model_state)
        
        # VERIFICATION: Re-evaluate the loaded best model to confirm metrics match training
        print(f"\n>>> Verifying loaded best model for fold {fold_idx + 1}...")
        verify_val_loss, verify_val_ap, verify_val_roc = evaluate(
            model, val_loader, criterion, device, use_mask=use_mask
        )
        verify_test_loss, verify_test_ap, verify_test_roc = evaluate(
            model, blind_test_loader, criterion, device, use_mask=use_mask
        )
        print(f">>> Verification (without threshold) - Val AP: {verify_val_ap:.4f}, Val ROC: {verify_val_roc:.4f}")
        print(f">>> Verification (without threshold) - Blind Test AP: {verify_test_ap:.4f}, Blind Test ROC: {verify_test_roc:.4f}")
        
        # Find best threshold on validation set
        best_threshold = find_best_threshold(
            model, val_loader, criterion, device, use_mask=use_mask
        )
        print(f"Best threshold for fold {fold_idx + 1}: {best_threshold:.4f}")
        
        # Get OOF predictions for this fold (residue-level)
        oof_preds, oof_labels, oof_loss = compute_predictions(
            model, val_loader, criterion, device, use_mask=use_mask
        )
        
        # Store OOF predictions (append residue-level predictions)
        all_oof_preds.append(oof_preds)
        all_oof_labels.append(oof_labels)
        
        # Evaluate on validation set with threshold
        val_loss, val_auc_pr, val_auc_roc, val_mcc, val_f1, val_acc = evaluate_w_threshold(
            model, val_loader, criterion, device, best_threshold, use_mask=use_mask
        )
        
        # Evaluate on blind test set
        test_loss, test_auc_pr, test_auc_roc, test_mcc, test_f1, test_acc = evaluate_w_threshold(
            model, blind_test_loader, criterion, device, best_threshold, use_mask=use_mask
        )
        
        # Check if metrics match the verification (they should be very close for AUC-PR/ROC)
        val_ap_diff = abs(val_auc_pr - verify_val_ap)
        test_ap_diff = abs(test_auc_pr - verify_test_ap)
        if val_ap_diff > 0.001 or test_ap_diff > 0.001:
            print(f"\n⚠️  WARNING: Metrics mismatch detected!")
            print(f"   Val AP difference: {val_ap_diff:.6f} (verify: {verify_val_ap:.4f} vs final: {val_auc_pr:.4f})")
            print(f"   Test AP difference: {test_ap_diff:.6f} (verify: {verify_test_ap:.4f} vs final: {test_auc_pr:.4f})")
        else:
            print(f"✓ Metrics verification passed (differences < 0.001)")
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Validation - Loss: {val_loss:.4f}, AP: {val_auc_pr:.4f}, ROC-AUC: {val_auc_roc:.4f}, MCC: {val_mcc:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        print(f"  Blind Test - Loss: {test_loss:.4f}, AP: {test_auc_pr:.4f}, ROC-AUC: {test_auc_roc:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
        
        # Save fold results
        fold_result = {
            'fold': fold_idx + 1,
            'threshold': float(best_threshold),
            'validation': {
                'loss': float(val_loss),
                'auc_pr': float(val_auc_pr),
                'auc_roc': float(val_auc_roc),
                'mcc': float(val_mcc),
                'f1': float(val_f1),
                'accuracy': float(val_acc)
            },
            'blind_test': {
                'loss': float(test_loss),
                'auc_pr': float(test_auc_pr),
                'auc_roc': float(test_auc_roc),
                'mcc': float(test_mcc),
                'f1': float(test_f1),
                'accuracy': float(test_acc)
            }
        }
        fold_results.append(fold_result)

        # Running uncertainty info for AUC-PR across completed folds (SEM)
        running_val_auc_pr = [f['validation']['auc_pr'] for f in fold_results]
        running_test_auc_pr = [f['blind_test']['auc_pr'] for f in fold_results]
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
            'fold': fold_idx + 1,
            'model_state_dict': best_model_state,
            'config': asdict(config),
            'threshold': float(best_threshold),
            'metrics': fold_result
        }
        
        fold_path = os.path.join(model_dir, f"fold_{fold_idx + 1}.pt")
        torch.save(checkpoint, fold_path)
        print(f"Saved fold {fold_idx + 1} checkpoint to {fold_path}")
    
    # STEP 4: Compute OOF metrics
    # OOF is computed on all residues from validation folds of all antigens
    # This gives an unbiased estimate of CV performance at the residue level
    print(f"\n{'='*80}")
    print("STEP 4: Computing Out-Of-Fold (OOF) Metrics")
    print("OOF = Predictions on validation folds of all 200 antigens (at residue level)")
    print(f"{'='*80}")
    
    # Verify that we have OOF predictions
    if len(all_oof_preds) == 0 or len(all_oof_labels) == 0:
        raise ValueError(f"No OOF predictions collected. all_oof_preds has {len(all_oof_preds)} elements, all_oof_labels has {len(all_oof_labels)} elements.")
    
    # Concatenate all OOF predictions from all folds
    all_oof_preds = np.concatenate(all_oof_preds)
    all_oof_labels = np.concatenate(all_oof_labels)
    
    print(f"Total OOF residues: {len(all_oof_preds)}")
    print(f"Positive labels: {np.sum(all_oof_labels)} ({100*np.mean(all_oof_labels):.2f}%)")
    print(f"Negative labels: {len(all_oof_labels) - np.sum(all_oof_labels)} ({100*(1-np.mean(all_oof_labels)):.2f}%)")
    
    # Verify we have both classes
    if len(np.unique(all_oof_labels)) < 2:
        raise ValueError(f"OOF labels contain only one class: {np.unique(all_oof_labels)}")
    
    # Compute OOF metrics with raw probabilities
    oof_auc_pr = average_precision_score(all_oof_labels, all_oof_preds)
    oof_auc_roc = roc_auc_score(all_oof_labels, all_oof_preds)
    
    # Find best threshold on OOF predictions
    best_oof_threshold = 0.5
    best_oof_mcc = -1.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        binary_preds = (all_oof_preds >= threshold).astype(int)
        try:
            mcc_score = matthews_corrcoef(all_oof_labels, binary_preds)
            if mcc_score > best_oof_mcc:
                best_oof_mcc = mcc_score
                best_oof_threshold = threshold
        except Exception as e:
            # Skip this threshold if MCC calculation fails
            continue
    
    print(f"\nBest OOF threshold found: {best_oof_threshold:.4f} with MCC: {best_oof_mcc:.4f}")
    
    # Compute final OOF metrics with best threshold
    oof_binary_preds = (all_oof_preds >= best_oof_threshold).astype(int)
    oof_mcc = matthews_corrcoef(all_oof_labels, oof_binary_preds)
    oof_f1 = f1_score(all_oof_labels, oof_binary_preds)
    oof_acc = accuracy_score(all_oof_labels, oof_binary_preds)
    
    print(f"\nOOF Metrics (threshold={best_oof_threshold:.4f}):")
    print(f"  AUC-PR: {oof_auc_pr:.4f}")
    print(f"  AUC-ROC: {oof_auc_roc:.4f}")
    print(f"  MCC: {oof_mcc:.4f}")
    print(f"  F1: {oof_f1:.4f}")
    print(f"  Accuracy: {oof_acc:.4f}")
    
    # Verify we have fold results
    if len(fold_results) == 0:
        raise ValueError("No fold results collected. All folds may have failed.")
    
    # Aggregate fold metrics
    def _mean(key_path):
        return np.mean([f[key_path[0]][key_path[1]] for f in fold_results])

    avg_val_metrics = {
        k: float(_mean(('validation', k)))
        for k in ('loss', 'auc_pr', 'auc_roc', 'mcc', 'f1', 'accuracy')
    }
    avg_test_metrics = {
        k: float(_mean(('blind_test', k)))
        for k in ('loss', 'auc_pr', 'auc_roc', 'mcc', 'f1', 'accuracy')
    }

    def _sem(key_path):
        values = [f[key_path[0]][key_path[1]] for f in fold_results]
        if len(values) < 2:
            return 0.0
        return np.std(values, ddof=1) / np.sqrt(len(values))

    sem_val_metrics = {
        k: float(_sem(('validation', k)))
        for k in ('loss', 'auc_pr', 'auc_roc', 'mcc', 'f1', 'accuracy')
    }
    sem_test_metrics = {
        k: float(_sem(('blind_test', k)))
        for k in ('loss', 'auc_pr', 'auc_roc', 'mcc', 'f1', 'accuracy')
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
    
    # Save OOF metrics
    oof_metrics = {
        'model_name': model_name,
        'num_folds': config.num_folds,
        'oof_threshold': float(best_oof_threshold),
        'oof_metrics': {
            'auc_pr': float(oof_auc_pr),
            'auc_roc': float(oof_auc_roc),
            'mcc': float(oof_mcc),
            'f1': float(oof_f1),
            'accuracy': float(oof_acc)
        },
        'average_validation_metrics': avg_val_metrics,
        'sem_validation_metrics': sem_val_metrics,
        'average_blind_test_metrics': avg_test_metrics,
        'sem_blind_test_metrics': sem_test_metrics,
        'fold_results': fold_results,
        'config': asdict(config)
    }
    
    oof_metrics_path = os.path.join(model_dir, "OOF_metrics.json")
    with open(oof_metrics_path, 'w') as f:
        json.dump(oof_metrics, f, indent=4)
    
    print(f"\nSaved OOF metrics to {oof_metrics_path}")
    
    return oof_metrics
