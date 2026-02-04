from __future__ import annotations

from dataclasses import dataclass
from logging import config
from typing import Iterable, Optional

from networkx import config
import numpy as np
import torch
from sklearn.metrics import average_precision_score
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


def _forward_batch(model, data, device: torch.device, use_edge_attr: bool):
	data = data.to(device)
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
			# Attempt to recompute edge_attr from node positions if available
			if hasattr(data, "pos") and data.pos is not None:
				print("[CORRECTION] Recomputing edge_attr from pos...")
				source_index = data.edge_index[0]
				target_index = data.edge_index[1]
				source_coords = data.pos[source_index]
				target_coords = data.pos[target_index]
				edge_attr = torch.norm(source_coords - target_coords, dim=1, keepdim=True).float()
				print(f"[DEBUG] edge_attr recomputed from pos: {tuple(edge_attr.shape)}")
			else:
				edge_attr = None
				print("[DEBUG] edge_attr dropped (no pos available)")
	# Check edge attr and edge index sizes again
	if edge_attr is not None and edge_attr.size(0) != data.edge_index.size(1):
		print(
			"[DEBUG] Final edge_attr/edge_index mismatch after recompute: "
			f"edge_attr={tuple(edge_attr.shape)}, edge_index={tuple(data.edge_index.shape)}"
		)
		edge_attr = None  # Drop edge_attr if still mismatched
	logits, edge_index_out, attn = model(data.x, data.edge_index, edge_attr=edge_attr)
	return logits, _prepare_targets(data.y), edge_index_out, attn


def train_one_epoch(
	model: torch.nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	device: torch.device,
	use_edge_attr: bool,
) -> float:
	model.train()
	total_loss = 0.0
	for data in loader:
		optimizer.zero_grad(set_to_none=True)
		logits, targets, edge_index_out, attn = _forward_batch(model, data, device, use_edge_attr)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
	model: torch.nn.Module,
	loader: DataLoader,
	criterion: torch.nn.Module,
	device: torch.device,
	use_edge_attr: bool,
):
	from sklearn.metrics import average_precision_score, roc_auc_score
	model.eval()
	total_loss = 0.0
	all_targets = []
	all_scores = []

	for data in loader:
		logits, targets, edge_index_out, attn = _forward_batch(model, data, device, use_edge_attr)
		loss = criterion(logits, targets)
		total_loss += loss.item()
		all_targets.append(targets.detach().cpu())
		all_scores.append(torch.sigmoid(logits.detach().cpu()))

	if all_targets:
		y_true = torch.cat(all_targets).numpy()
		y_score = torch.cat(all_scores).numpy()
		auc_pr = average_precision_score(y_true, y_score)
		auc_roc = roc_auc_score(y_true, y_score)
	else:
		auc_pr = float("nan")
		auc_roc = float("nan")
	return total_loss / max(1, len(loader)), auc_pr, auc_roc


def infer_edge_dim(dataset) -> Optional[int]:
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
	device: torch.device,
	use_edge_attr: bool,
	n_epochs: int,
	patience: int,
	log_file: Optional[str] = None,
) -> Iterable[float]:
	best_ap = -1.0
	best_loss = float("inf")
	best_model_state = model.state_dict()
	current_patience = patience
	
	# Create log file if specified
	if log_file is None:
		import datetime
		log_file = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
	
	with open(log_file, 'w') as f:
		f.write("Epoch,Patience,Train_Loss,Val_Loss,Train_AUC_PR,Val_AUC_PR\n")
	
	for epoch in range(1, n_epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_edge_attr)
		train_eval_loss, train_auc_pr, train_auc_roc = evaluate(model, train_loader, criterion, device, use_edge_attr)
		val_loss, val_auc_pr, val_auc_roc = evaluate(model, val_loader, criterion, device, use_edge_attr)

		if np.isfinite(val_auc_pr) and val_auc_pr > best_ap:
			best_ap = val_auc_pr
			best_model_state = model.state_dict()
		if val_loss < best_loss:
			best_loss = val_loss

		# Write metrics to log file
		with open(log_file, 'a') as f:
			f.write(f"{epoch},{patience - current_patience},{train_loss:.4f},{val_loss:.4f},{train_auc_pr:.4f},{val_auc_pr:.4f}\n")

		print(
			f"Epoch {epoch}/{n_epochs} | "
			f"Train Loss: {train_loss:.4f} | Train AP: {train_auc_pr:.4f} | "
			f"Val Loss: {val_loss:.4f} | Val AP: {val_auc_pr:.4f} | "
			f"patience: {patience - current_patience} / {patience}"
		)

		# Early stopping check
		if val_loss < best_loss and val_auc_pr < best_ap:
			current_patience -= 1
			if current_patience == 0:
				print("Early stopping triggered.")
				return best_model_state
		else:
			current_patience = patience
	return best_model_state


from sklearn.model_selection import KFold
def run_cross_validation(
	dataset,
	model_builder,
	config: TrainingConfig,
	device: torch.device,
	use_edge_attr: bool,
	seed: int = 42,):

	models = []
	indices = np.arange(len(dataset))
	kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=seed)

	fold_metrics = []
	for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices), start=1):
		train_subset = [dataset[i] for i in train_idx]
		val_subset = [dataset[i] for i in val_idx]

		train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
		val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

		model = model_builder().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
		criterion = torch.nn.BCEWithLogitsLoss()

		best_ap = -1.0
		best_model_state = train_n_epochs(
			model,
			train_loader,
			val_loader,
			optimizer,
			criterion,
			device,
			use_edge_attr,
			config.epochs,
		)

		model.load_state_dict(best_model_state)
		checkpoint = {
			'model_state_dict': model.state_dict(),
			'config': config,
			}

		torch.save(checkpoint, f'best_model_fold_{fold_idx}.pt')
		models.append(model)
		fold_metrics.append(best_ap)
		print(f"Fold {fold_idx} best AP: {best_ap:.4f}")
	
	mean_ap = float(np.mean(fold_metrics)) if fold_metrics else float("nan")
	std_ap = float(np.std(fold_metrics)) if fold_metrics else float("nan")
	print(f"Cross-validation AP: {mean_ap:.4f} ± {std_ap:.4f}")
	return fold_metrics



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
	for data in val_loader:
		logits, targets, edge_index_out, attn = _forward_batch(model, data, device, use_edge_attr)
		all_targets.append(targets.detach().cpu())
		all_scores.append(torch.sigmoid(logits.detach().cpu()))
	y_true = torch.cat(all_targets).numpy()
	y_scores = torch.cat(all_scores).numpy()
	best_threshold = 0.5
	best_mcc = -1.0
	for threshold in np.arange(0.0, 1.01, 0.01):
		pred_labels = (y_scores >= threshold).astype(int)
		mcc = matthews_corrcoef(y_true, pred_labels)
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
	from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score, accuracy_score
	model.eval()
	total_loss = 0.0
	all_targets = []
	all_scores = []

	for data in loader:
		logits, targets, edge_index_out, attn = _forward_batch(model, data, device, use_edge_attr)
		loss = criterion(logits, targets)
		total_loss += loss.item()
		all_targets.append(targets.detach().cpu())
		all_scores.append(torch.sigmoid(logits.detach().cpu()))

	if all_targets:
		y_true = torch.cat(all_targets).numpy()
		y_score = torch.cat(all_scores).numpy()
		auc_pr = average_precision_score(y_true, y_score)
		auc_roc = roc_auc_score(y_true, y_score)
		mcc = matthews_corrcoef(y_true, (y_score >= threshold).astype(int))
		acc = accuracy_score(y_true, (y_score >= threshold).astype(int))
		f1_score_val = f1_score(y_true, (y_score >= threshold).astype(int))
	else:
		auc_pr = float("nan")
		auc_roc = float("nan")
	return total_loss / max(1, len(loader)), auc_pr, auc_roc, mcc, acc, f1_score_val