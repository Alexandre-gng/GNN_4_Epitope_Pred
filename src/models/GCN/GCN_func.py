from typing import Iterable, Optional
import copy
import numpy as np

import torch
import torch.nn as nn

from dataclasses import dataclass, asdict
from torch_geometric.loader import DataLoader

from GCN import PyGGCNModel


@dataclass
class TrainingConfig:
    """Training configuration for the GCN model."""
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-8
    num_folds: int = 10
    n_hidden: int = 128
    n_layers: int = 2
    n_edge_attr: int = 1
    dropout: float = 0.5
    patience: int = 30
    alpha: float = 0.25       # Focal loss alpha
    gamma: float = 2.0        # Focal loss gamma
    best_threshold: Optional[float] = None


def train_one_epoch(model, train_loader, optimizer, criterion, device, use_mask=False):
    """
    Train the GCN model for one epoch on the given training data.

    Args:
        model (nn.Module): The GCN model to be trained.
        train_loader (Iterable): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function to optimize.
        device (torch.device): Device to run the training on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues during training.
    Returns:
        avg_loss (float): Average loss over the training set for this epoch.
    """
    model.train()
    total_loss = 0
    num_samples = 0
    skipped_nonfinite_loss_batches = 0
    skipped_empty_batches = 0
    for batch_graphs in train_loader:
        batch_graphs = batch_graphs.to(device)

        optimizer.zero_grad()
        out = model(batch_graphs)
        out = out.squeeze(-1)  # Flatten output from [N, 1] to [N]

        # Apply mask if provided: mask=True means ignore that residue
        # The data may store the mask as 'mask' or 'train_mask'
        _mask_attr = None
        if use_mask:
            if hasattr(batch_graphs, 'mask') and batch_graphs.mask is not None:
                _mask_attr = batch_graphs.mask
            elif hasattr(batch_graphs, 'train_mask') and batch_graphs.train_mask is not None:
                _mask_attr = batch_graphs.train_mask

        if _mask_attr is not None:
            mask = _mask_attr.bool()
            out_masked = out[~mask]
            y_masked = batch_graphs.y[~mask].float()
        else:
            out_masked = out
            y_masked = batch_graphs.y.float()

        if out_masked.numel() == 0:
            skipped_empty_batches += 1
            continue

        y_masked = torch.nan_to_num(y_masked, nan=0.0, posinf=1.0, neginf=0.0)

        loss = criterion(out_masked, y_masked)
        if not torch.isfinite(loss):
            skipped_nonfinite_loss_batches += 1
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_samples += 1

    if skipped_empty_batches > 0 or skipped_nonfinite_loss_batches > 0:
        print(
            f"[train_one_epoch] skipped batches: empty={skipped_empty_batches}, "
            f"nonfinite_loss={skipped_nonfinite_loss_batches}"
        )

    return total_loss / num_samples if num_samples > 0 else 0


def train_n_epochs(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    blind_test_loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
    use_mask: bool = True,
):
    """
    Train the GCN model for a specified number of epochs, evaluating on validation and test sets.

    Args:
        model (nn.Module): The GCN model to be trained.
        train_loader (Iterable): DataLoader for the training set.
        val_loader (Iterable): DataLoader for the validation set.
        blind_test_loader (Iterable): DataLoader for the blind test set.
        criterion (nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        config (TrainingConfig): Configuration parameters for training.
        device (torch.device): Device to run the training on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues during training and evaluation.
    Returns:
        best_state_dict: State dict of the best model (highest Val AUC-PR).
    """
    best_val_loss = float('inf')
    best_ap = -1.0
    best_epoch = 0
    best_val_ap_at_best = -1.0
    best_test_ap_at_best = -1.0
    current_patience = config.patience
    best_state_dict = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_mask=use_mask)
        train_eval_loss, train_auc_pr, train_auc_roc = evaluate(model, train_loader, criterion, device, use_mask=use_mask)
        val_loss, val_auc_pr, val_auc_roc = evaluate(model, val_loader, criterion, device, use_mask=use_mask)
        blind_test_loss, blind_test_auc_pr, blind_test_auc_roc = evaluate(model, blind_test_loader, criterion, device, use_mask=use_mask)

        # Early stopping check based on validation AUC-PR
        if np.isfinite(val_auc_pr) and val_auc_pr > best_ap:
            best_ap = val_auc_pr
            best_epoch = epoch
            best_val_ap_at_best = val_auc_pr
            best_test_ap_at_best = blind_test_auc_pr
            best_state_dict = copy.deepcopy(model.state_dict())
            current_patience = config.patience
        else:
            current_patience -= 1
            if current_patience == 0:
                print("Early stopping triggered.")
                break

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"patience: {config.patience - current_patience} / {config.patience} | "
            f"Train Loss: {train_loss:.4f} | Train AP: {train_auc_pr:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AP: {val_auc_pr:.4f} | "
            f"Blind Test Loss: {blind_test_loss:.4f} | Blind Test AP: {blind_test_auc_pr:.4f}"
        )

    # Return best model if found, otherwise return current state
    if best_state_dict is None:
        print("Warning: No improvement observed during training. Returning current model state.")
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        print(
            f"\n>>> Best model from epoch {best_epoch} "
            f"with Val AP: {best_val_ap_at_best:.4f}, Blind Test AP: {best_test_ap_at_best:.4f}"
        )

    return best_state_dict


def evaluate(model, data_loader, criterion, device, use_mask=False):
    """Evaluate the GCN model on a given dataset, calculating loss, AUC-PR, and AUC-ROC.

    Args:
        model (nn.Module): The GCN model to be evaluated.
        data_loader (Iterable): DataLoader for the dataset to evaluate on.
        criterion (nn.Module): Loss function to calculate the loss.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues during evaluation.
    Returns:
        avg_loss (float): Average loss over the dataset.
        auc_pr (float): Area Under the Precision-Recall Curve.
        auc_roc (float): Area Under the Receiver Operating Characteristic Curve.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()

    all_preds, all_labels, avg_loss = compute_predictions(model, data_loader, criterion, device, use_mask)

    if all_preds.size == 0 or all_labels.size == 0:
        return float('inf'), 0.0, 0.0

    all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=1.0, neginf=0.0)
    all_labels = np.nan_to_num(all_labels, nan=0.0, posinf=1.0, neginf=0.0).astype(int)

    try:
        auc_pr = average_precision_score(all_labels, all_preds)
    except ValueError:
        auc_pr = 0.0

    try:
        auc_roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_roc = 0.0
    return avg_loss, auc_pr, auc_roc


def evaluate_w_threshold(model, data_loader, criterion, device, threshold, use_mask=False):
    """Evaluate the GCN model on a given dataset using a specified threshold for binary classification.

    Args:
        model (nn.Module): The GCN model to be evaluated.
        data_loader (Iterable): DataLoader for the dataset to evaluate on.
        criterion (nn.Module): Loss function to calculate the loss.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        threshold (float): Threshold for converting predicted probabilities to binary labels.
        use_mask (bool): Whether to use mask to filter residues during evaluation.
    Returns:
        avg_loss (float): Average loss over the dataset.
        auc_pr (float): Area Under the Precision-Recall Curve.
        auc_roc (float): Area Under the Receiver Operating Characteristic Curve.
        mcc (float): Matthews Correlation Coefficient.
        f1 (float): F1 Score.
        acc (float): Accuracy.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
    model.eval()
    total_loss = 0
    all_probs = []
    all_binary_preds = []
    all_labels = []
    num_samples = 0

    with torch.no_grad():
        for batch_graphs in data_loader:
            batch_graphs = batch_graphs.to(device)
            out = model(batch_graphs)
            out = out.squeeze(-1)

            # Apply mask if provided (check both 'mask' and 'train_mask')
            _mask_attr = None
            if use_mask:
                if hasattr(batch_graphs, 'mask') and batch_graphs.mask is not None:
                    _mask_attr = batch_graphs.mask
                elif hasattr(batch_graphs, 'train_mask') and batch_graphs.train_mask is not None:
                    _mask_attr = batch_graphs.train_mask

            if _mask_attr is not None:
                mask = _mask_attr.bool()
                out_masked = out[~mask]
                y_masked = batch_graphs.y[~mask].float()
            else:
                out_masked = out
                y_masked = batch_graphs.y.float()

            if out_masked.numel() == 0:
                continue

            y_masked = torch.nan_to_num(y_masked, nan=0.0, posinf=1.0, neginf=0.0)

            loss = criterion(out_masked, y_masked)
            if not torch.isfinite(loss):
                continue
            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(out_masked)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            all_probs.append(probs.cpu())
            binary_preds = (probs >= threshold).long()
            all_binary_preds.append(binary_preds.cpu())
            all_labels.append(y_masked.cpu())
            num_samples += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 0

    if len(all_probs) == 0 or len(all_labels) == 0:
        return float('inf'), 0.0, 0.0, 0.0, 0.0, 0.0

    # Convert to numpy arrays
    all_probs_np = torch.cat(all_probs).numpy()
    all_binary_preds_np = torch.cat(all_binary_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy().astype(int)

    all_probs_np = np.nan_to_num(all_probs_np, nan=0.0, posinf=1.0, neginf=0.0)
    all_labels_np = np.nan_to_num(all_labels_np, nan=0.0, posinf=1.0, neginf=0.0).astype(int)

    # AUC-PR and AUC-ROC need raw probabilities
    try:
        auc_pr = average_precision_score(all_labels_np, all_probs_np)
    except ValueError:
        auc_pr = 0.0

    try:
        auc_roc = roc_auc_score(all_labels_np, all_probs_np)
    except ValueError:
        auc_roc = 0.0
    # MCC, F1, and Accuracy need binary predictions
    mcc = matthews_corrcoef(all_labels_np, all_binary_preds_np)
    f1 = f1_score(all_labels_np, all_binary_preds_np)
    acc = accuracy_score(all_labels_np, all_binary_preds_np)

    return avg_loss, auc_pr, auc_roc, mcc, f1, acc


def compute_predictions(model, data_loader, criterion, device, use_mask=False):
    """Compute predictions from the GCN model on a given dataset.

    Args:
        model (nn.Module): The GCN model to compute predictions with.
        data_loader (Iterable): DataLoader for the dataset to compute predictions on.
        criterion (nn.Module): Loss function to calculate the loss.
        device (torch.device): Device to run the computation on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues during computation.
    Returns:
        all_preds (numpy.ndarray): Array of predicted probabilities for all samples in the dataset.
        all_labels (numpy.ndarray): Array of true labels for all samples in the dataset.
        avg_loss (float): Average loss over the dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch_graphs in data_loader:
            batch_graphs = batch_graphs.to(device)
            out = model(batch_graphs)
            out = out.squeeze(-1)

            # Apply mask if provided (check both 'mask' and 'train_mask')
            _mask_attr = None
            if use_mask:
                if hasattr(batch_graphs, 'mask') and batch_graphs.mask is not None:
                    _mask_attr = batch_graphs.mask
                elif hasattr(batch_graphs, 'train_mask') and batch_graphs.train_mask is not None:
                    _mask_attr = batch_graphs.train_mask

            if _mask_attr is not None:
                mask = _mask_attr.bool()
                out_masked = out[~mask]
                y_masked = batch_graphs.y[~mask].float()
            else:
                out_masked = out
                y_masked = batch_graphs.y.float()

            if out_masked.numel() == 0:
                continue

            y_masked = torch.nan_to_num(y_masked, nan=0.0, posinf=1.0, neginf=0.0)

            loss = criterion(out_masked, y_masked)
            if not torch.isfinite(loss):
                continue
            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(out_masked)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            all_preds.append(probs.cpu())
            all_labels.append(y_masked.cpu())
            num_samples += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 0

    if len(all_preds) == 0 or len(all_labels) == 0:
        return np.array([]), np.array([]), float('inf')

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=1.0, neginf=0.0)
    all_labels = np.nan_to_num(all_labels, nan=0.0, posinf=1.0, neginf=0.0)

    return all_preds, all_labels, avg_loss


def find_best_threshold(model, data_loader, criterion, device, use_mask=False):
    """
    Find best threshold by maximizing the MCC on the validation set.

    Args:
        model (nn.Module): The GCN model to be evaluated.
        data_loader (Iterable): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function (unused but kept for consistency).
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues when finding threshold.
    Returns:
        best_threshold (float): The threshold that yields the best MCC on the validation set.
    """
    from sklearn.metrics import matthews_corrcoef
    model.eval()

    all_preds, all_labels, _ = compute_predictions(model, data_loader, criterion, device, use_mask)

    if all_preds.size == 0 or all_labels.size == 0:
        return 0.5

    best_threshold = 0.5
    best_mcc = -1.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        binary_preds = (all_preds >= threshold).astype(int)
        mcc_score = matthews_corrcoef(all_labels, binary_preds)
        if mcc_score > best_mcc:
            best_mcc = mcc_score
            best_threshold = threshold
    return best_threshold
