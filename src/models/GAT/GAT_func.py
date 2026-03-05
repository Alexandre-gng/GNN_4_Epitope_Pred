from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import torch
from torch_geometric.loader import DataLoader


@dataclass
class TrainingConfig:
	epochs: int = 50
	batch_size: int = 8
	learning_rate: float = 1e-5
	weight_decay: float = 1e-8
	num_folds: int = 10
	hidden_dim: int = 128
	heads: int = 8
	num_layers: int = 8
	dropout: float = 0.3
	patience: int = 30
	best_threshold: Optional[float] = None


def _prepare_targets(y: torch.Tensor) -> torch.Tensor:
	if y is None:
		raise ValueError("Graph data is missing labels (data.y).")
	if y.dim() > 1:
		y = y.view(-1)
	return y.float()


def _get_node_features(data) -> torch.Tensor:
	x = getattr(data, "x", None)
	if isinstance(x, torch.Tensor):
		return x

	if hasattr(data, "keys") and "node_attrs" in data:
		node_attrs = data["node_attrs"]
		if isinstance(node_attrs, torch.Tensor):
			return node_attrs
		if callable(node_attrs):
			owner = getattr(node_attrs, "__self__", None)
			if owner is not None and hasattr(owner, "keys") and "node_attrs" in owner:
				owner_node_attrs = owner["node_attrs"]
				if isinstance(owner_node_attrs, torch.Tensor):
					return owner_node_attrs

	available_keys = list(data.keys()) if hasattr(data, "keys") else []
	raise ValueError(
		"No valid node feature tensor found in graph data. "
		f"Available keys: {available_keys}"
	)


def _safe_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
	from sklearn.metrics import average_precision_score, roc_auc_score

	if y_true.size == 0:
		return 0.0, 0.0

	auc_pr = 0.0
	auc_roc = 0.0
	try:
		auc_pr = average_precision_score(y_true, y_score)
	except ValueError:
		pass
	try:
		auc_roc = roc_auc_score(y_true, y_score)
	except ValueError:
		pass
	return float(auc_pr), float(auc_roc)


def _forward_batch(model, data, device: torch.device, use_edge_attr: bool):
	data = data.to(device)
	x = _get_node_features(data)
	edge_attr = data.edge_attr if use_edge_attr and hasattr(data, "edge_attr") else None
	if edge_attr is not None:
		if edge_attr.dim() == 1:
			edge_attr = edge_attr.view(-1, 1)
		# Guard against mismatched edge_attr length vs edge_index edges
		if edge_attr.size(0) != data.edge_index.size(1):
			print(
				"[DEBUG] edge_attr/edge_index mismatch: "
				f"edge_attr={tuple(edge_attr.shape)}, edge_index={tuple(data.edge_index.shape)}, "
				f"num_nodes={getattr(data, 'num_nodes', 'NA')}, pos={getattr(data, 'pos', None) is not None}"
			)
			edge_attr = None

	# Check edge attr and edge index sizes again
	if edge_attr is not None and edge_attr.size(0) != data.edge_index.size(1):
		print(
			"[DEBUG] Final edge_attr/edge_index mismatch after recompute: "
			f"edge_attr={tuple(edge_attr.shape)}, edge_index={tuple(data.edge_index.shape)}"
		)
		edge_attr = None

	logits, edge_index_out, attn = model(x, data.edge_index, edge_attr=edge_attr)
	return logits, _prepare_targets(data.y), edge_index_out, attn


def train_one_epoch(
	model: torch.nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	device: torch.device,
	use_edge_attr: bool,
) -> float:
	"""
	Train the model for one epoch and compute the average loss.
	Args:
		model: The GNN model to train
		loader: DataLoader for training data
		optimizer: Optimizer for updating model parameters
		criterion: Loss function to compute training loss
		device: Computation device (CPU or GPU)
		use_edge_attr: Whether to use edge attributes during training
	Returns:
		Average training loss for the epoch
	"""
	model.train()
	total_loss = 0.0
	num_samples = 0
	for data in loader:
		optimizer.zero_grad(set_to_none=True)
		logits, targets, _, _ = _forward_batch(model, data, device, use_edge_attr)
		loss = criterion(logits, targets)
		if not torch.isfinite(loss):
			continue
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		total_loss += loss.item()
		num_samples += 1
	return total_loss / max(1, num_samples)


@torch.no_grad()
def compute_predictions(
	model: torch.nn.Module,
	loader: DataLoader,
	criterion: torch.nn.Module,
	device: torch.device,
	use_edge_attr: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
	model.eval()
	total_loss = 0.0
	num_samples = 0
	all_targets = []
	all_scores = []

	for data in loader:
		logits, targets, _, _ = _forward_batch(model, data, device, use_edge_attr)
		loss = criterion(logits, targets)
		if not torch.isfinite(loss):
			continue
		total_loss += loss.item()
		num_samples += 1
		all_targets.append(targets.detach().cpu())
		all_scores.append(torch.sigmoid(logits.detach().cpu()))

	if not all_targets:
		return np.array([]), np.array([]), float("inf")

	y_true = torch.cat(all_targets).numpy()
	y_score = torch.cat(all_scores).numpy()
	return y_score, y_true, total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate(
	model: torch.nn.Module,
	loader: DataLoader,
	criterion: torch.nn.Module,
	device: torch.device,
	use_edge_attr: bool,
):
	"""
	Evaluate the model on the given data loader and compute loss, AUC-PR, and AUC-ROC.

	Args:
	model: Trained model to evaluate
	loader: DataLoader for evaluation data
	criterion: Loss function to compute average loss
	device: Computation device (CPU or GPU)
	use_edge_attr: Whether to use edge attributes during evaluation
	Returns:
	Average loss, AUC-PR, and AUC-ROC
	"""
	y_score, y_true, avg_loss = compute_predictions(model, loader, criterion, device, use_edge_attr)
	if y_true.size == 0:
		return float("inf"), 0.0, 0.0
	auc_pr, auc_roc = _safe_auc_metrics(y_true, y_score)
	return avg_loss, auc_pr, auc_roc


def infer_edge_dim(dataset) -> Optional[int]:
	"""
	Infers the edge attribute dimension from the dataset.
	Returns:
		edge_dim (int): The dimension of edge attributes, or None if not found.
	"""
	for data in dataset:
		if hasattr(data, "edge_attr") and data.edge_attr is not None:
			if data.edge_attr.dim() == 1:
				return 1
			return data.edge_attr.size(-1)
	return None


def train_n_epochs(
	model: torch.nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	name_model: str,
	device: torch.device,
	use_edge_attr: bool,
	n_epochs: int,
	patience: int,
	use_mlflow: bool = True,
) -> Iterable[float]:
	mlflow = None
	if use_mlflow:
		try:
			import mlflow
			mlflow.set_experiment(name_model)
		except Exception as e:
			print(f"Warning: Could not set up mlflow tracking: {e}")
			use_mlflow = False

	best_ap = -1.0
	best_loss = float("inf")
	best_model_state = copy.deepcopy(model.state_dict())
	current_patience = patience

	for epoch in range(1, n_epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_edge_attr)
		_, train_auc_pr, train_auc_roc = evaluate(model, train_loader, criterion, device, use_edge_attr)
		val_loss, val_auc_pr, val_auc_roc = evaluate(model, val_loader, criterion, device, use_edge_attr)

		# Log metrics to mlflow
		if use_mlflow:
			try:
				mlflow.log_metric("train_loss", train_loss, step=epoch)
				mlflow.log_metric("train_auc_pr", train_auc_pr, step=epoch)
				mlflow.log_metric("train_auc_roc", train_auc_roc, step=epoch)
				mlflow.log_metric("val_loss", val_loss, step=epoch)
				mlflow.log_metric("val_auc_pr", val_auc_pr, step=epoch)
				mlflow.log_metric("val_auc_roc", val_auc_roc, step=epoch)
			except Exception as e:
				print(f"Warning: Could not log metrics to mlflow: {e}")

		# Early stopping check based on validation AUC-PR
		if np.isfinite(val_auc_pr) and val_auc_pr > best_ap:
			best_ap = val_auc_pr
			best_model_state = copy.deepcopy(model.state_dict())
			current_patience = patience  # Reset patience on improvement
		else:
			current_patience -= 1
			if current_patience == 0:
				print("Early stopping triggered.")
				return best_model_state
		
		if val_loss < best_loss:
			best_loss = val_loss

		print(
			f"Epoch {epoch}/{n_epochs} | "
			f"Train Loss: {train_loss:.4f} | Train AP: {train_auc_pr:.4f} | "
			f"Val Loss: {val_loss:.4f} | Val AP: {val_auc_pr:.4f} | "
			f"patience: {patience - current_patience} / {patience}"
		)
	return best_model_state



def find_best_threshold(model, val_loader, device, use_edge_attr) -> float:
	"""
	Find the threshold that maximizes the MCC.

	Args:
		y_true: Ground truth binary labels [num_samples]
		y_scores: Predicted scores/probabilities [num_samples]

	Returns:
		Best threshold value
	"""
	from sklearn.metrics import matthews_corrcoef
	model.eval()
	all_targets = []
	all_scores = []
	with torch.no_grad():
		for data in val_loader:
			logits, targets, _, _ = _forward_batch(model, data, device, use_edge_attr)
			all_targets.append(targets.detach().cpu())
			all_scores.append(torch.sigmoid(logits.detach().cpu()))

	if not all_targets:
		return 0.5

	y_true = torch.cat(all_targets).numpy()
	y_scores = torch.cat(all_scores).numpy()
	best_threshold = 0.5
	best_mcc = -1.0
	for threshold in np.arange(0.0, 1.01, 0.01):
		pred_labels = (y_scores >= threshold).astype(int)
		try:
			mcc = matthews_corrcoef(y_true, pred_labels)
		except Exception:
			continue
		if mcc > best_mcc:
			best_mcc = mcc
			best_threshold = threshold
	return best_threshold


def evaluate_w_threshold(
	model: torch.nn.Module,
	loader: DataLoader,
	threshold: float,
	criterion: torch.nn.Module,
	device: torch.device,
	use_edge_attr: bool,
) -> tuple[float, float, float, float, float, float]:
	"""
	Evaluate the model using a predefined threshold for classification.

	Args:
		model: Trained model
		loader: DataLoader for evaluation data
		criterion: Loss function
		device: Computation device
		use_edge_attr: Whether to use edge attributes

	Returns:
		Tuple of (loss, AUC-PR, AUC-ROC, MCC, Accuracy, F1 Score)
	"""
	from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
	model.eval()

	y_score, y_true, avg_loss = compute_predictions(model, loader, criterion, device, use_edge_attr)
	if y_true.size == 0:
		return float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0

	auc_pr, auc_roc = _safe_auc_metrics(y_true, y_score)
	pred = (y_score >= threshold).astype(int)
	try:
		mcc = matthews_corrcoef(y_true, pred)
	except Exception:
		mcc = 0.0
	try:
		acc = accuracy_score(y_true, pred)
	except Exception:
		acc = 0.0
	try:
		f1 = f1_score(y_true, pred)
	except Exception:
		f1 = 0.0

	return avg_loss, auc_pr, auc_roc, float(mcc), float(acc), float(f1)