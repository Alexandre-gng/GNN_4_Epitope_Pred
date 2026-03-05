from typing import Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

import numpy as np
import os
import copy

from EGNN import EGNN


@dataclass
class TrainingConfig:
    """Training configuration for the EGNN model."""
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    num_folds: int = 10
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 32
    edge_dim: int = 2
    dropout: float = 0.1
    patience: int = 10
    alpha: float = 0.25       # Focal loss alpha
    gamma: float = 2.0        # Focal loss gamma
    update_coords: bool = True
    max_coord_step: Optional[float] = 1.0
    max_abs_coord_value: Optional[float] = 1e4
    best_threshold: Optional[float] = None


def _recompute_edge_attr_from_coords(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    reference_edge_attr: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Recompute edge attributes from updated coordinates.

    If reference_edge_attr is provided, the returned tensor keeps the same number
    of columns as the reference tensor to stay compatible with the model input.
    """
    rel_coords = coords[edge_index[0]] - coords[edge_index[1]]
    distances = torch.norm(rel_coords, dim=1, keepdim=True)
    inv_sq = 1.0 / (distances.pow(2) + 1e-6)

    if reference_edge_attr is None:
        return torch.cat([distances, inv_sq], dim=1)

    if reference_edge_attr.dim() != 2:
        return distances

    ref_dim = reference_edge_attr.shape[1]

    if ref_dim <= 1:
        return distances
    if ref_dim == 2:
        return torch.cat([distances, inv_sq], dim=1)

    updated = reference_edge_attr.clone()
    updated[:, 0:1] = distances
    updated[:, 1:2] = inv_sq
    return updated


def _select_logits_and_targets(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch,
    device,
    use_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Select supervised nodes according to mask semantics.

    Supported semantics:
    - mask_mode == "exclude": True means ignore node.
    - mask_mode == "include": True means keep node.

    Returns:
        logits, targets, used_fallback_all_nodes
    """
    logits = features.reshape(-1)
    targets = labels.float().reshape(-1)

    if not use_mask:
        return logits, targets, False

    if "mask" not in batch or batch["mask"] is None:
        return logits, targets, False

    mask = batch["mask"].to(device).bool().reshape(-1)
    if mask.numel() != logits.numel():
        return logits, targets, True

    mask_mode = batch.get("mask_mode", "exclude")
    if mask_mode == "include":
        selected = mask
    else:
        selected = ~mask

    if selected.any():
        return logits[selected], targets[selected], False

    return logits, targets, True


def compute_predictions(model, data_loader, criterion, device, use_mask=False):
    """
    Compute predictions and loss for a given model and data loader.

    Args:
        model: The EGNN model to evaluate
        data_loader: DataLoader providing batches of graph data
        criterion: Loss function to compute loss
        device: Device to run computations on (e.g., 'cuda' or 'cpu')
        use_mask: Whether to apply masking to ignore certain residues in loss computation
    
    Returns:
        all_preds (numpy.ndarray): Array of predicted probabilities for all samples in the dataset.
        all_labels (numpy.ndarray): Array of true labels for all samples in the dataset.
        avg_loss (float): Average loss over the dataset.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_nodes = 0
    skipped_empty_batches = 0
    skipped_nonfinite_loss_batches = 0
    mask_fallback_batches = 0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            total_batches += 1
            # Move data to device
            node_attr = batch["node_attrs"].to(device)
            coords = batch["coords"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device) if "edge_attr" in batch and batch["edge_attr"] is not None else None
            labels = batch["y"].to(device)
            
            # Forward pass
            features, coords_updated = model(node_attr, coords, edge_index, edge_attr)

            logits, targets, used_mask_fallback = _select_logits_and_targets(
                features, labels, batch, device, use_mask
            )
            if used_mask_fallback:
                mask_fallback_batches += 1

            if logits.numel() == 0:
                skipped_empty_batches += 1
                continue
    
            # Compute loss (squeeze predictions to match label shape)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                skipped_nonfinite_loss_batches += 1
                continue
            
            # Accumulate metrics
            total_loss += loss.item() * logits.numel()
            total_nodes += logits.numel()

            # Replace NaN/Inf in predictions to avoid downstream crashes
            preds_np = logits.cpu().numpy()
            preds_np = np.nan_to_num(preds_np, nan=0.0, posinf=0.0, neginf=0.0)
            all_predictions.append(preds_np)
            all_labels.append(targets.cpu().numpy())

    avg_loss = total_loss / total_nodes if total_nodes > 0 else 0.0

    if total_batches > 0 and (skipped_empty_batches > 0 or skipped_nonfinite_loss_batches > 0):
        skipped_total = skipped_empty_batches + skipped_nonfinite_loss_batches
        print(
            f"[compute_predictions] skipped {skipped_total}/{total_batches} batches "
            f"({100.0 * skipped_total / total_batches:.1f}%) "
            f"| empty_mask={skipped_empty_batches}, nonfinite_loss={skipped_nonfinite_loss_batches}"
        )

    if total_batches > 0 and mask_fallback_batches > 0:
        print(
            f"[compute_predictions] mask fallback to all nodes on "
            f"{mask_fallback_batches}/{total_batches} batches "
            f"({100.0 * mask_fallback_batches / total_batches:.1f}%)"
        )

    if len(all_predictions) == 0 or len(all_labels) == 0:
        return np.array([]), np.array([]), float('inf')
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_predictions, all_labels, avg_loss




def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_mask=True,
    update_coords=True,
    batch_size=1,
    max_coord_step: float | None = 1.0,
    max_abs_coord_value: float | None = 1e3,
):
    """
    Train the model for one epoch.
    
    Args:
        model: The EGNN model to train
        dataloader: DataLoader providing batches of graph data
        optimizer: Optimizer for gradient updates
        criterion: Loss function
        device: Device to run computations on
        use_mask: Whether to apply masking to ignore certain residues
        update_coords: Whether to persist updated coordinates and edge attributes
        max_coord_step: Maximum per-node coordinate displacement. If None, no displacement clipping is applied.
        max_abs_coord_value: Maximum absolute coordinate value. If None, absolute clipping is disabled.
    
    Returns:
        avg_loss (float): Average training loss over the epoch
    """
    model.train()
    total_loss = 0.0
    total_nodes = 0
    batch_size = max(1, int(batch_size))
    skipped_empty_batches = 0
    skipped_nonfinite_loss_batches = 0
    mask_fallback_batches = 0
    total_batches = 0

    optimizer.zero_grad()
    samples_in_step = 0

    for batch_idx, batch in enumerate(dataloader):
        total_batches += 1
        # Move data to device
        node_attr = batch["node_attrs"].to(device)
        coords = batch["coords"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device) if "edge_attr" in batch and batch["edge_attr"] is not None else None
        labels = batch["y"].to(device)
        
        # Forward pass
        features, coords_updated = model(node_attr, coords, edge_index, edge_attr)

        # Cordinate and edge attribute updates, mandatory for EGNN
        if update_coords:
            with torch.no_grad():
                previous_coords_cpu = batch["coords"]
                if torch.is_tensor(previous_coords_cpu):
                    previous_coords_cpu = previous_coords_cpu.detach().cpu()

                updated_coords_cpu = coords_updated.detach().cpu()

                if torch.isfinite(updated_coords_cpu).all() and max_coord_step is not None:
                    delta = updated_coords_cpu - previous_coords_cpu
                    delta_norm = torch.norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
                    step_scale = torch.clamp(max_coord_step / delta_norm, max=1.0)
                    updated_coords_cpu = previous_coords_cpu + delta * step_scale

                if max_abs_coord_value is not None:
                    updated_coords_cpu = torch.nan_to_num(
                        updated_coords_cpu,
                        nan=0.0,
                        posinf=max_abs_coord_value,
                        neginf=-max_abs_coord_value,
                    ).clamp(-max_abs_coord_value, max_abs_coord_value)
                else:
                    finite_mask = torch.isfinite(updated_coords_cpu)
                    updated_coords_cpu = torch.where(finite_mask, updated_coords_cpu, previous_coords_cpu)

                batch["coords"] = updated_coords_cpu
                if "edge_index" in batch and batch["edge_index"] is not None:
                    edge_index_cpu = batch["edge_index"]
                    if torch.is_tensor(edge_index_cpu):
                        edge_index_cpu = edge_index_cpu.detach().cpu()
                    current_edge_attr_cpu = None
                    if "edge_attr" in batch and batch["edge_attr"] is not None:
                        current_edge_attr_cpu = batch["edge_attr"]
                        if torch.is_tensor(current_edge_attr_cpu):
                            current_edge_attr_cpu = current_edge_attr_cpu.detach().cpu()
                    batch["edge_attr"] = _recompute_edge_attr_from_coords(
                        edge_index_cpu,
                        updated_coords_cpu,
                        reference_edge_attr=current_edge_attr_cpu,
                    )

        logits, targets, used_mask_fallback = _select_logits_and_targets(
            features, labels, batch, device, use_mask
        )
        if used_mask_fallback:
            mask_fallback_batches += 1

        if logits.numel() == 0:
            skipped_empty_batches += 1
            continue

        # Compute loss (squeeze predictions to match label shape)
        loss = criterion(logits, targets)
        
        # Skip backward pass if loss is NaN/Inf (prevents weight corruption)
        if not torch.isfinite(loss):
            skipped_nonfinite_loss_batches += 1
            optimizer.zero_grad()
            samples_in_step = 0
            continue

        # Backward pass and optimization (logical mini-batching over graph samples)
        (loss / batch_size).backward()
        samples_in_step += 1
        is_step_boundary = samples_in_step >= batch_size or batch_idx == len(dataloader) - 1
        if is_step_boundary:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            samples_in_step = 0
        
        # Accumulate metrics
        total_loss += loss.item() * logits.numel()
        total_nodes += logits.numel()

    avg_loss = total_loss / total_nodes if total_nodes > 0 else 0.0

    if total_batches > 0 and (skipped_empty_batches > 0 or skipped_nonfinite_loss_batches > 0):
        skipped_total = skipped_empty_batches + skipped_nonfinite_loss_batches
        print(
            f"[train_one_epoch] skipped {skipped_total}/{total_batches} batches "
            f"({100.0 * skipped_total / total_batches:.1f}%) "
            f"| empty_mask={skipped_empty_batches}, nonfinite_loss={skipped_nonfinite_loss_batches}"
        )

    if total_batches > 0 and mask_fallback_batches > 0:
        print(
            f"[train_one_epoch] mask fallback to all nodes on "
            f"{mask_fallback_batches}/{total_batches} batches "
            f"({100.0 * mask_fallback_batches / total_batches:.1f}%)"
        )

    return avg_loss





def train_n_epochs(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    num_epochs=100,
    patience_max=10,
    use_mask=False,
    update_coords=False,
    batch_size=1,
    max_coord_step: float | None = 1.0,
    max_abs_coord_value: float | None = 1e3,
    trial=None,
    prune_metric="auc_pr",
    return_best_auc_pr=False,
):
    """
    Train the model for multiple epochs with early stopping.
    
    Args:
        model: The EGNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for gradient updates
        criterion: Loss function
        device: Device to run computations on
        num_epochs: Maximum number of epochs to train
        patience_max: Maximum number of epochs without improvement before early stopping
        use_mask: Whether to apply masking to ignore certain residues
        update_coords: Whether to persist updated coordinates and edge attributes during training
        max_coord_step: Maximum per-node coordinate displacement. If None, no displacement clipping is applied.
        max_abs_coord_value: Maximum absolute coordinate value. If None, absolute clipping is disabled.
    
    Returns:
        best_state_dict: State dictionary of the best model
        best_val_loss: Validation loss of the best model
    """
    best_val_loss = float('inf')
    best_auc_pr = 0.0
    patience = 0
    best_state_dict = None
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_mask,
            update_coords,
            batch_size=batch_size,
            max_coord_step=max_coord_step,
            max_abs_coord_value=max_abs_coord_value,
        )
        val_preds, val_labels, val_loss = compute_predictions(model, val_loader, criterion, device, use_mask)
        test_preds, test_labels, test_loss = compute_predictions(model, test_loader, criterion, device, use_mask)
        print(f"{'='*80}")

        metrics_val = evaluate(model, val_loader, criterion, device, use_mask)
        metrics_test = evaluate(model, test_loader, criterion, device, use_mask)
        print(f"VAL AUC-ROC: {metrics_val['auc_roc']:.4f}, VAL AUC-PR: {metrics_val['auc_pr']:.4f}")
        print(f"Test AUC-ROC: {metrics_test['auc_roc']:.4f}, Test AUC-PR: {metrics_test['auc_pr']:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} --- train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f}, best_val_loss: {best_val_loss:.4f}, patience: {patience}/{patience_max}")

        # Save best model
        if (
            metrics_val['auc_pr'] > best_auc_pr
            or (metrics_val['auc_pr'] == best_auc_pr and val_loss < best_val_loss)
        ):
            best_val_loss = val_loss
            best_auc_pr = metrics_val['auc_pr']
            patience = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience += 1

        if trial is not None:
            if prune_metric == "val_loss":
                trial.report(val_loss, epoch)
            else:
                trial.report(metrics_val['auc_pr'], epoch)

            if trial.should_prune():
                raise RuntimeError("OPTUNA_PRUNED")

        if patience >= patience_max:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    if return_best_auc_pr:
        return best_state_dict, best_val_loss, best_auc_pr
    return best_state_dict, best_val_loss



def evaluate(model, data_loader, criterion, device, use_mask=False):
    """
    Returns different metrics: loss, accuracy, precision, recall, auc_pr, auc_roc for the given model and data loader.

    Args:
        model: The EGNN model to evaluate
        data_loader: DataLoader providing batches of graph data
        criterion: Loss function to compute loss
        device: Device to run computations on (e.g., 'cuda' or 'cpu')
        use_mask: Whether to apply masking to ignore certain residues in loss computation
    Returns:
        metrics (dict): Dictionary containing loss, accuracy, precision, recall, auc_pr, auc_roc
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

    all_predictions, all_labels, avg_loss = compute_predictions(model, data_loader, criterion, device, use_mask)

    if all_predictions.size == 0 or all_labels.size == 0:
        return {"loss": float('inf'), "auc_roc": 0.0, "auc_pr": 0.0}

    # Guard against NaN predictions (e.g. from gradient explosion)
    if np.isnan(all_predictions).any():
        print("WARNING: NaN detected in predictions during evaluate(). Returning zero metrics.")
        return {"loss": float('inf'), "auc_roc": 0.0, "auc_pr": 0.0}

    # Compute metrics
    auc_roc = roc_auc_score(all_labels, all_predictions)
    auc_pr = average_precision_score(all_labels, all_predictions)

    metrics = {
        "loss": avg_loss,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr
    }

    return metrics


def evaluate_w_threshold(model, data_loader, criterion, device, threshold, use_mask=False):
    """Evaluate the EGNN model on a given dataset using a specified threshold for binary classification.

    Args:
        model: The EGNN model to be evaluated.
        data_loader: DataLoader providing batches of graph data.
        criterion: Loss function to calculate the loss.
        device: Device to run the evaluation on (CPU or GPU).
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
    total_loss = 0.0
    all_probs = []
    all_binary_preds = []
    all_labels = []
    total_nodes = 0
    skipped_empty_batches = 0
    skipped_nonfinite_loss_batches = 0
    mask_fallback_batches = 0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            total_batches += 1
            # Move data to device
            node_attr = batch["node_attrs"].to(device)
            coords = batch["coords"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device) if "edge_attr" in batch and batch["edge_attr"] is not None else None
            labels = batch["y"].to(device)

            # Forward pass
            features, coords_updated = model(node_attr, coords, edge_index, edge_attr)

            logits, targets, used_mask_fallback = _select_logits_and_targets(
                features, labels, batch, device, use_mask
            )
            if used_mask_fallback:
                mask_fallback_batches += 1

            if logits.numel() == 0:
                skipped_empty_batches += 1
                continue

            # Compute loss (squeeze predictions to match label shape)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                skipped_nonfinite_loss_batches += 1
                continue
            total_loss += loss.item() * logits.numel()
            total_nodes += logits.numel()

            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs = probs.reshape(-1)
            all_probs.append(probs.cpu())
            binary_preds = (probs >= threshold).long()
            all_binary_preds.append(binary_preds.cpu())
            all_labels.append(targets.reshape(-1).cpu())

    avg_loss = total_loss / total_nodes if total_nodes > 0 else 0.0

    if total_batches > 0 and (skipped_empty_batches > 0 or skipped_nonfinite_loss_batches > 0):
        skipped_total = skipped_empty_batches + skipped_nonfinite_loss_batches
        print(
            f"[evaluate_w_threshold] skipped {skipped_total}/{total_batches} batches "
            f"({100.0 * skipped_total / total_batches:.1f}%) "
            f"| empty_mask={skipped_empty_batches}, nonfinite_loss={skipped_nonfinite_loss_batches}"
        )

    if total_batches > 0 and mask_fallback_batches > 0:
        print(
            f"[evaluate_w_threshold] mask fallback to all nodes on "
            f"{mask_fallback_batches}/{total_batches} batches "
            f"({100.0 * mask_fallback_batches / total_batches:.1f}%)"
        )

    if len(all_probs) == 0 or len(all_labels) == 0:
        return float('inf'), 0.0, 0.0, 0.0, 0.0, 0.0

    # Convert to numpy arrays
    all_probs_np = torch.cat(all_probs).numpy()
    all_binary_preds_np = torch.cat(all_binary_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()

    all_probs_np = np.nan_to_num(all_probs_np, nan=0.0, posinf=1.0, neginf=0.0)
    all_labels_np = np.nan_to_num(all_labels_np, nan=0.0, posinf=1.0, neginf=0.0)
    all_labels_np = (all_labels_np > 0.5).astype(int)

    # AUC-PR and AUC-ROC need raw probabilities
    auc_pr = average_precision_score(all_labels_np, all_probs_np)
    auc_roc = roc_auc_score(all_labels_np, all_probs_np)
    # MCC, F1, and Accuracy need binary predictions
    mcc = matthews_corrcoef(all_labels_np, all_binary_preds_np)
    f1 = f1_score(all_labels_np, all_binary_preds_np)
    acc = accuracy_score(all_labels_np, all_binary_preds_np)

    return avg_loss, auc_pr, auc_roc, mcc, f1, acc


def find_best_threshold(model, data_loader, criterion, device, use_mask=False):
    """
    Find best threshold by maximizing the MCC on the validation set.

    Args:
        model: The EGNN model to be evaluated.
        data_loader: DataLoader for the validation dataset.
        criterion: Loss function (unused but kept for consistency).
        device: Device to run the evaluation on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues when finding threshold.
    Returns:
        best_threshold (float): The threshold that yields the best MCC on the validation set.
    """
    from sklearn.metrics import matthews_corrcoef
    model.eval()

    all_preds, all_labels, _ = compute_predictions(model, data_loader, criterion, device, use_mask)

    if all_preds.size == 0 or all_labels.size == 0:
        return 0.5

    # Apply sigmoid to convert logits to probabilities
    all_probs = 1.0 / (1.0 + np.exp(-all_preds.reshape(-1)))

    best_threshold = 0.5
    best_mcc = -1.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        binary_preds = (all_probs >= threshold).astype(int)
        mcc_score = matthews_corrcoef(all_labels, binary_preds)
        if mcc_score > best_mcc:
            best_mcc = mcc_score
            best_threshold = threshold
    return best_threshold