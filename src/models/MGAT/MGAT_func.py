from typing import Iterable, Optional
import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from MGAT import MultiView_GAT


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int, view_idx: int, context: str = ""):
    """Fail fast with a readable error instead of a CUDA device-side assert."""
    if edge_index.numel() == 0:
        return
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"Invalid edge_index shape for view {view_idx}: got {tuple(edge_index.shape)}. {context}"
        )
    if edge_index.dtype != torch.long:
        raise ValueError(
            f"Invalid edge_index dtype for view {view_idx}: got {edge_index.dtype}, expected torch.long. {context}"
        )
    min_idx = int(edge_index.min().item())
    max_idx = int(edge_index.max().item())
    if min_idx < 0 or max_idx >= int(num_nodes):
        raise ValueError(
            f"edge_index out of bounds for view {view_idx}: min={min_idx}, max={max_idx}, num_nodes={int(num_nodes)}. {context}"
        )


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-8
    num_folds: int = 10
    hidden_dim: int = 128
    edge_dim: int = 1
    heads: int = 8
    num_layers: int = 8
    dropout: float = 0.3
    reg_lambda: float = 4e-3 # regularization strength for the diversity term, can be tuned
    patience: int = 30
    alpha: float = 0.25
    gamma: float = 2.0
    best_threshold: Optional[float] = None




def train_one_epoch(model, train_loader, optimizer, criterion, device, use_mask=False):
    """
    Train the MGAT model for one epoch on the given training data.

    Args:
        model (nn.Module): The MGAT model to be trained.
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
    for batch_idx, batch_graphs in enumerate(train_loader):
        # batch_graphs is a Batch object (batched Data objects)
        batch_graphs = batch_graphs.to(device)
        
        optimizer.zero_grad()
        # Extract edge indices for all views
        edge_indices_list = [batch_graphs[f'edge_index_{i}'] for i in range(model.num_views)]
        # Extract edge attributes for all views only if the model uses them
        if model.edge_dim is not None and model.edge_dim > 0:
            edge_attrs_list = [batch_graphs[f'edge_attr_{i}'] if f'edge_attr_{i}' in batch_graphs else None for i in range(model.num_views)]
        else:
            edge_attrs_list = None

        num_nodes = int(batch_graphs.num_nodes)
        for i, ei in enumerate(edge_indices_list):
            _validate_edge_index(ei, num_nodes=num_nodes, view_idx=i, context="(train)")
        out, _ = model(batch_graphs['node_attrs'], edge_indices_list, edge_attrs=edge_attrs_list)
        out = out.squeeze(-1)  # Flatten output from [N, 1] to [N]
        
        # Apply mask if provided: mask=True means ignore that residue
        if use_mask and hasattr(batch_graphs, 'mask') and batch_graphs.mask is not None:
            mask = batch_graphs.mask.bool()
            out_masked = out[~mask]
            y_masked = batch_graphs.y[~mask].float()
        else:
            out_masked = out
            y_masked = batch_graphs.y.float()
        
        loss = model.get_total_loss(out_masked, y_masked, criterion)  # Total loss includes regularization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_samples += 1
    
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
    use_mask: bool = True
):
    """
    Train the MGAT model for a specified number of epochs, evaluating on validation and test sets.

    Args:
        model (nn.Module): The MGAT model to be trained.
        train_loader (Iterable): DataLoader for the training set.
        val_loader (Iterable): DataLoader for the validation set.
        blind_test_loader (Iterable): DataLoader for the blind test set.
        criterion (nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        config (TrainingConfig): Configuration parameters for training.
        device (torch.device): Device to run the training on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues during training and evaluation.
    """
    import copy
    
    best_val_loss = float('inf')
    best_ap = -1.0
    best_epoch = 0
    best_val_ap_at_best = -1.0
    best_test_ap_at_best = -1.0
    current_patience = config.patience
    best_state_dict = None  # Will be set on first improvement

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
            # CRITICAL FIX: Make a deep copy to preserve the exact model state
            best_state_dict = copy.deepcopy(model.state_dict())
            current_patience = config.patience  # Reset patience on improvement
        else:
            current_patience -= 1
            if current_patience == 0:
                print("Early stopping triggered.")
                break
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"Epoch {epoch}/{config.epochs} | patience: {config.patience - current_patience} / {config.patience} | "
            f"Train Loss: {train_loss:.4f} | Train AP: {train_auc_pr:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AP: {val_auc_pr:.4f} | "
            f"Blind Test Loss: {blind_test_loss:.4f} | Blind Test AP: {blind_test_auc_pr:.4f} | "
            f"patience: {config.patience - current_patience} / {config.patience}"
        )
      
    # Return best model if found, otherwise return current state
    if best_state_dict is None:
        print("Warning: No improvement observed during training. Returning current model state.")
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        print(f"\n>>> Best model from epoch {best_epoch} with Val AP: {best_val_ap_at_best:.4f}, Blind Test AP: {best_test_ap_at_best:.4f}")
    
    return best_state_dict



def evaluate(model, data_loader, criterion, device, use_mask=False):
    """Evaluate the MGAT model on a given dataset, calculating loss, AUC-PR, and AUC-ROC.

    Args:
        model (nn.Module): The MGAT model to be evaluated.
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

    auc_pr = average_precision_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_preds)
    # Calculate AUC-PR and AUC-ROC
    return avg_loss, auc_pr, auc_roc



def evaluate_w_threshold(model, data_loader, criterion, device, threshold, use_mask=False):
    """Evaluate the MGAT model on a given dataset using a specified threshold for binary classification.

    Args:
        model (nn.Module): The MGAT model to be evaluated.
        data_loader (Iterable): DataLoader for the dataset to evaluate on.
        criterion (nn.Module): Loss function to calculate the loss.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        threshold (float): Threshold for converting predicted probabilities to binary labels.
        use_mask (bool): Whether to use mask to filter residues during evaluation.
    Returns:
        avg_loss (float): Average loss over the dataset.
        auc_pr (float): Area Under the Precision-Recall Curve.
        auc_roc (float): Area Under the Receiver Operating Characteristic Curve.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
    model.eval()
    total_loss = 0
    all_probs = []  # Store raw probabilities for AUC-PR and AUC-ROC
    all_binary_preds = []  # Store binary predictions for other metrics
    all_labels = []
    num_samples = 0
    with torch.no_grad():
        for batch_graphs in data_loader:
            batch_graphs = batch_graphs.to(device)
            edge_indices_list = [batch_graphs[f'edge_index_{i}'] for i in range(model.num_views)]
            if model.edge_dim is not None and model.edge_dim > 0:
                edge_attrs_list = [batch_graphs[f'edge_attr_{i}'] if f'edge_attr_{i}' in batch_graphs else None for i in range(model.num_views)]
            else:
                edge_attrs_list = None
            out, view_weights = model(batch_graphs['node_attrs'], edge_indices_list, edge_attrs=edge_attrs_list)
            out = out.squeeze(-1)  # Flatten output from [N, 1] to [N]
            
            # Apply mask if provided: mask=True means ignore that residue
            if use_mask and hasattr(batch_graphs, 'mask') and batch_graphs.mask is not None:
                mask = batch_graphs.mask.bool()
                out_masked = out[~mask]
                y_masked = batch_graphs.y[~mask].float()
            else:
                out_masked = out
                y_masked = batch_graphs.y.float()
            
            loss = model.get_total_loss(out_masked, y_masked, criterion)  # Total loss includes regularization
            total_loss += loss.item()
            
            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(out_masked)
            # Store raw probabilities for AUC metrics
            all_probs.append(probs.cpu())
            # Store binary predictions (as integers) for other metrics
            binary_preds = (probs >= threshold).long()
            all_binary_preds.append(binary_preds.cpu())
            all_labels.append(y_masked.cpu())
            num_samples += 1
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0

    # Convert to numpy arrays
    all_probs_np = torch.cat(all_probs).numpy()
    all_binary_preds_np = torch.cat(all_binary_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy().astype(int)
    
    # AUC-PR and AUC-ROC need raw probabilities
    auc_pr = average_precision_score(all_labels_np, all_probs_np)
    auc_roc = roc_auc_score(all_labels_np, all_probs_np)
    # MCC, F1, and Accuracy need binary predictions
    mcc = matthews_corrcoef(all_labels_np, all_binary_preds_np)
    f1 = f1_score(all_labels_np, all_binary_preds_np)
    acc = accuracy_score(all_labels_np, all_binary_preds_np)

    return avg_loss, auc_pr, auc_roc, mcc, f1, acc



def compute_predictions(model, data_loader, criterion, device, use_mask=False):
    """Compute predictions from the MGAT model on a given dataset.

    Args:
        model (nn.Module): The MGAT model to compute predictions with.
        data_loader (Iterable): DataLoader for the dataset to compute predictions on.
        criterion (nn.Module): Loss function to calculate the loss.
        device (torch.device): Device to run the computation on (CPU or GPU).
        use_edge_attr (bool): Whether to use edge attributes in the model input.
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
            edge_indices_list = [batch_graphs[f'edge_index_{i}'] for i in range(model.num_views)]
            if model.edge_dim is not None and model.edge_dim > 0:
                edge_attrs_list = [batch_graphs[f'edge_attr_{i}'] if f'edge_attr_{i}' in batch_graphs else None for i in range(model.num_views)]
            else:
                edge_attrs_list = None
            out, _ = model(batch_graphs['node_attrs'], edge_indices_list, edge_attrs=edge_attrs_list)
            out = out.squeeze(-1)  # Flatten output from [N, 1] to [N]

            # Apply mask if provided: mask=True means ignore that residue
            if use_mask and hasattr(batch_graphs, 'mask') and batch_graphs.mask is not None:
                mask = batch_graphs.mask.bool()
                out_masked = out[~mask]
                y_masked = batch_graphs.y[~mask].float()
            else:
                out_masked = out
                y_masked = batch_graphs.y.float()

            loss = model.get_total_loss(out_masked, y_masked, criterion)  # Total loss includes regularization
            total_loss += loss.item()
            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(out_masked)
            all_preds.append(probs.cpu())
            all_labels.append(y_masked.cpu())
            num_samples += 1
            
    avg_loss = total_loss / num_samples if num_samples > 0 else 0

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_preds, all_labels, avg_loss



def find_best_threshold(model, data_loader, criterion, device, use_edge_attr=False, use_mask=False):
    """
    Find best threshold by maximizing the MCC on the validation set.
    Args:
        model (nn.Module): The MGAT model to be evaluated.
        data_loader (Iterable): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function (unused but kept for consistency).
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        use_edge_attr (bool): Whether to use edge attributes in the model input.
        use_mask (bool): Whether to use mask to filter residues when finding threshold.
    Returns:
        best_threshold (float): The threshold that yields the best MCC on the validation set.
    """
    from sklearn.metrics import matthews_corrcoef
    model.eval()
    
    all_preds, all_labels, _ = compute_predictions(model, data_loader, criterion, device, use_mask)
    
    best_threshold = 0.5
    best_mcc = -1.0  # MCC can be negative, so start with -1
    for threshold in np.arange(0.0, 1.01, 0.01):
        binary_preds = (all_preds >= threshold).astype(int)
        mcc_score = matthews_corrcoef(all_labels, binary_preds)
        if mcc_score > best_mcc:
            best_mcc = mcc_score
            best_threshold = threshold
    return best_threshold



def get_attn_by_view(model, data_loader, device, use_mask=False):
    """Extract average attention weights per view and per class from the MGAT model.

    Args:
        model (nn.Module): The MGAT model to extract attention weights from.
        data_loader (Iterable): DataLoader for the dataset to extract attention weights on.
        device (torch.device): Device to run the extraction on (CPU or GPU).
        use_mask (bool): Whether to use mask to filter residues during extraction.
    Returns:
        result (dict): Dictionary mapping class labels to average attention weights per view.
                       Format: {class_label: array of shape [num_views]}
                       Also returns overall average per view if return_overall_avg is True.
    """
    model.eval()
    all_view_weights = []
    all_labels = []
    
    with torch.no_grad():
        for batch_graphs in data_loader:
            batch_graphs = batch_graphs.to(device)
            edge_indices_list = [batch_graphs[f'edge_index_{i}'] for i in range(model.num_views)]
            if model.edge_dim is not None and model.edge_dim > 0:
                edge_attrs_list = [batch_graphs[f'edge_attr_{i}'] if f'edge_attr_{i}' in batch_graphs else None for i in range(model.num_views)]
            else:
                edge_attrs_list = None
            global_embedding, view_weights = model(batch_graphs['node_attrs'], edge_indices_list, edge_attrs=edge_attrs_list)
            
            # Apply mask if provided: mask=True means ignore that residue
            if use_mask and hasattr(batch_graphs, 'mask') and batch_graphs.mask is not None:
                mask = batch_graphs.mask.bool()
                view_weights = view_weights[~mask]
                labels = batch_graphs.y[~mask]
            else:
                labels = batch_graphs.y
            
            all_view_weights.append(view_weights.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_view_weights = torch.cat(all_view_weights, dim=0)  # [total_nodes, num_views]
    all_labels = torch.cat(all_labels, dim=0)  # [total_nodes]
    
    # Compute average attention per view and per class
    unique_classes = torch.unique(all_labels)
    avg_attn_per_class = {}
    
    for cls in unique_classes:
        cls_mask = (all_labels == cls)
        cls_weights = all_view_weights[cls_mask]
        avg_weights = cls_weights.mean(dim=0)  # Average across nodes of this class
        avg_attn_per_class[int(cls.item())] = avg_weights.numpy()
    
    # Also compute overall average attention per view across all nodes
    overall_avg_per_view = all_view_weights.mean(dim=0).numpy()
    avg_attn_per_class['overall'] = overall_avg_per_view
    
    return avg_attn_per_class