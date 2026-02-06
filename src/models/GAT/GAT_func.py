from __future__ import annotations

from dataclasses import dataclass, asdict
from logging import config
from typing import Iterable, Optional
import os
from pathlib import Path

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
			edge_attr = None  # Drop edge_attr if mismatched
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
	best_ap = -1.0
	best_loss = float("inf")
	best_model_state = model.state_dict()
	current_patience = patience
	

	for epoch in range(1, n_epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_edge_attr)
		train_eval_loss, train_auc_pr, train_auc_roc = evaluate(model, train_loader, criterion, device, use_edge_attr)
		val_loss, val_auc_pr, val_auc_roc = evaluate(model, val_loader, criterion, device, use_edge_attr)

		# Log metrics to mlflow
		if use_mlflow:
			try:
				import mlflow
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
			best_model_state = model.state_dict()
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


from sklearn.model_selection import KFold
def run_cross_validation(
	dataset,
	test_dataset,
	model_builder,
	criterion,
	config: TrainingConfig,
	device: torch.device,
	use_edge_attr: bool,
	model_name: str,
	seed: int = 42,):
	"""
	Run k-fold cross-validation with MLflow tracking and Out-Of-fold metrics calculation.
	The threshold is calculated after training each fold by maximizing MCC.
	Then evaluates on test dataset.
	
	Args:
		dataset: Full dataset to cross-validate
		test_dataset: Test dataset for final evaluation
		model_builder: Function to build a new model instance
		criterion: Loss function
		config: Training configuration
		device: torch device for computation
		use_edge_attr: Whether to use edge attributes
		model_name: Name of the model (for directory and MLflow experiment)
		seed: Random seed
	"""
	import mlflow
	from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score, accuracy_score
	
	# Setup directory structure
	model_dir = Path("src/models/trained_model") / model_name / "cross_val"
	model_dir.mkdir(parents=True, exist_ok=True)
	
	# MLflow setup - store in src/mlruns
	mlflow_dir = Path("src/mlruns").absolute()
	mlflow.set_tracking_uri(f"file:{mlflow_dir}")
	mlflow.set_experiment(f"CrossValidation_{model_name}")
	
	# Initialize OOF predictions and targets
	oof_predictions = {}
	oof_targets = {}
	all_fold_metrics = []
	all_fold_thresholds = []
	all_test_predictions = []  # Store test predictions from each fold
	
	indices = np.arange(len(dataset))
	kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=seed)

	for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices), start=1):
		with mlflow.start_run(run_name=f"Fold_{fold_idx}"):
			train_subset = [dataset[i] for i in train_idx]
			val_subset = [dataset[i] for i in val_idx]

			train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
			val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

			model = model_builder().to(device)
			optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

			# Train model with fold-specific log file
			best_model_state = train_n_epochs(
				model,
				train_loader,
				val_loader,
				optimizer,
				criterion,
				f"{model_name}_fold_{fold_idx}",
				device,
				use_edge_attr,
				config.epochs,
				config.patience,
				use_mlflow=True,
			)

			# Load best model for this fold
			model.load_state_dict(best_model_state)
			
			# Find best threshold for this fold by maximizing MCC
			print(f"\nFinding best threshold for fold {fold_idx}...")
			fold_threshold = find_best_threshold(model, val_loader=val_loader, device=device, use_edge_attr=use_edge_attr)
			print(f"Best threshold for fold {fold_idx}: {fold_threshold:.4f}")
			all_fold_thresholds.append(fold_threshold)
			
			# Evaluate fold with the computed threshold
			fold_loss, fold_auc_pr, fold_auc_roc, fold_mcc, fold_acc, fold_f1 = evaluate_w_threshold(
				model, val_loader, fold_threshold, criterion, device, use_edge_attr
			)
			
			# Log fold final metrics to MLflow
			mlflow.log_metric("final_loss", fold_loss)
			mlflow.log_metric("final_auc_pr", fold_auc_pr)
			mlflow.log_metric("final_auc_roc", fold_auc_roc)
			mlflow.log_metric("final_mcc", fold_mcc)
			mlflow.log_metric("final_accuracy", fold_acc)
			mlflow.log_metric("final_f1_score", fold_f1)
			mlflow.log_metric("best_threshold", fold_threshold)
			
			# Store fold metrics
			fold_metrics_dict = {
				"loss": fold_loss,
				"auc_pr": fold_auc_pr,
				"auc_roc": fold_auc_roc,
				"mcc": fold_mcc,
				"accuracy": fold_acc,
				"f1_score": fold_f1,
				"threshold": fold_threshold,
			}
			all_fold_metrics.append(fold_metrics_dict)
			
			# Collect OOF predictions and targets for this fold
			model.eval()
			fold_oof_preds = []
			fold_oof_targets = []
			with torch.no_grad():
				for data in val_loader:
					logits, targets, _, _ = _forward_batch(model, data, device, use_edge_attr)
					fold_oof_preds.append(torch.sigmoid(logits).detach().cpu())
					fold_oof_targets.append(targets.detach().cpu())
			
			fold_oof_preds = torch.cat(fold_oof_preds, dim=0).numpy()
			fold_oof_targets = torch.cat(fold_oof_targets, dim=0).numpy()
			
			# Store OOF predictions with validation indices
			for idx_in_val, val_sample_idx in enumerate(val_idx):
				if idx_in_val < len(fold_oof_preds):
					oof_predictions[int(val_sample_idx)] = fold_oof_preds[idx_in_val]
					oof_targets[int(val_sample_idx)] = fold_oof_targets[idx_in_val]
			
			# Save fold model with threshold in config
			checkpoint = {
				'model_state_dict': model.state_dict(),
				'config': asdict(config),
				'fold': fold_idx,
				'metrics': fold_metrics_dict,
				'threshold': fold_threshold,
			}
			fold_model_path = model_dir / f"fold_{fold_idx}_model.pt"
			torch.save(checkpoint, fold_model_path)
			
			# Get predictions on test dataset for this fold
			test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
			fold_test_preds = []
			with torch.no_grad():
				for data in test_loader:
					logits, _, _, _ = _forward_batch(model, data, device, use_edge_attr)
					fold_test_preds.append(torch.sigmoid(logits).detach().cpu())
			fold_test_preds = torch.cat(fold_test_preds, dim=0).numpy()
			all_test_predictions.append(fold_test_preds)
			
			print(f"Fold {fold_idx}/{config.num_folds} | Loss: {fold_loss:.4f} | AUC-PR: {fold_auc_pr:.4f} | "
				  f"AUC-ROC: {fold_auc_roc:.4f} | MCC: {fold_mcc:.4f} | Acc: {fold_acc:.4f} | F1: {fold_f1:.4f}")
	
	# Calculate Out-Of-Fold metrics
	print("\n" + "="*80)
	print("Calculating Out-Of-Fold Metrics...")
	
	# Compile OOF predictions and targets in order
	oof_indices = sorted(oof_predictions.keys())
	oof_preds = np.array([oof_predictions[idx] for idx in oof_indices])
	oof_trues = np.array([oof_targets[idx] for idx in oof_indices])
	
	# Calculate average threshold across folds
	avg_threshold = np.mean(all_fold_thresholds)
	print(f"Average threshold across folds: {avg_threshold:.4f}")
	
	# Calculate OOF metrics with average threshold
	oof_auc_pr = average_precision_score(oof_trues, oof_preds)
	oof_auc_roc = roc_auc_score(oof_trues, oof_preds)
	oof_mcc = matthews_corrcoef(oof_trues, (oof_preds >= avg_threshold).astype(int))
	oof_acc = accuracy_score(oof_trues, (oof_preds >= avg_threshold).astype(int))
	oof_f1 = f1_score(oof_trues, (oof_preds >= avg_threshold).astype(int))
	
	# Log final OOF metrics to MLflow
	with mlflow.start_run(run_name="OOF_Summary"):
		mlflow.log_metric("oof_auc_pr", oof_auc_pr)
		mlflow.log_metric("oof_auc_roc", oof_auc_roc)
		mlflow.log_metric("oof_mcc", oof_mcc)
		mlflow.log_metric("oof_accuracy", oof_acc)
		mlflow.log_metric("oof_f1_score", oof_f1)
		mlflow.log_param("avg_threshold", avg_threshold)
		for i, thresh in enumerate(all_fold_thresholds, start=1):
			mlflow.log_param(f"fold_{i}_threshold", thresh)
	
	# Save OOF metrics to file
	oof_metrics_path = Path("src/models/trained_model") / model_name / "OOF_metrics.txt"
	oof_metrics_path.parent.mkdir(parents=True, exist_ok=True)
	
	with open(oof_metrics_path, 'w') as f:
		f.write(f"Out-Of-Fold Metrics for Model: {model_name}\n")
		f.write(f"Number of Folds: {config.num_folds}\n")
		f.write(f"Average Classification Threshold: {avg_threshold:.4f}\n")
		f.write("="*80 + "\n\n")
		f.write("Fold-wise Metrics:\n")
		f.write("-"*80 + "\n")
		for i, fold_metrics_dict in enumerate(all_fold_metrics, start=1):
			f.write(f"Fold {i}:\n")
			f.write(f"  Threshold: {fold_metrics_dict['threshold']:.4f}\n")
			f.write(f"  AUC_PR: {fold_metrics_dict['auc_pr']:.4f}\n")
			f.write(f"  AUC_ROC: {fold_metrics_dict['auc_roc']:.4f}\n")
			f.write(f"  Accuracy: {fold_metrics_dict['accuracy']:.4f}\n")
			f.write(f"  F1_epitope: {fold_metrics_dict['f1_score']:.4f}\n")
			f.write(f"  MCC: {fold_metrics_dict['mcc']:.4f}\n")
			f.write("\n")
		
		f.write("-"*80 + "\n")
		f.write("Overall Out-Of-Fold Metrics:\n")
		f.write(f"  AUC_PR: {oof_auc_pr:.4f}\n")
		f.write(f"  AUC_ROC: {oof_auc_roc:.4f}\n")
		f.write(f"  Accuracy: {oof_acc:.4f}\n")
		f.write(f"  F1_epitope: {oof_f1:.4f}\n")
		f.write(f"  MCC: {oof_mcc:.4f}\n")
	
	print(f"\nOut-Of-Fold Results:")
	print(f"  AUC_PR: {oof_auc_pr:.4f}")
	print(f"  AUC_ROC: {oof_auc_roc:.4f}")
	print(f"  Accuracy: {oof_acc:.4f}")
	print(f"  F1_epitope: {oof_f1:.4f}")
	print(f"  MCC: {oof_mcc:.4f}")
	print(f"\nMetrics saved to: {oof_metrics_path}")
	print(f"Models saved to: {model_dir}")
	print("="*80 + "\n")
	
	# === TEST DATASET EVALUATION ===
	print("="*80)
	print("Evaluating on Test Dataset...")
	print("="*80)
	
	# Calculate mean predictions across all folds
	test_preds_array = np.array(all_test_predictions)  # Shape: (num_folds, num_test_samples)
	test_preds_mean = np.mean(test_preds_array, axis=0)  # Average across folds
	
	# Get test targets (handle both node-level and graph-level labels)
	test_targets = []
	for data in test_dataset:
		if data.y.dim() > 1:
			test_targets.extend(data.y.view(-1).cpu().numpy())
		else:
			test_targets.extend(data.y.cpu().numpy())
	test_targets = np.array(test_targets)
	
	# Calculate test metrics with average threshold
	test_auc_pr = average_precision_score(test_targets, test_preds_mean)
	test_auc_roc = roc_auc_score(test_targets, test_preds_mean)
	test_mcc = matthews_corrcoef(test_targets, (test_preds_mean >= avg_threshold).astype(int))
	test_acc = accuracy_score(test_targets, (test_preds_mean >= avg_threshold).astype(int))
	test_f1 = f1_score(test_targets, (test_preds_mean >= avg_threshold).astype(int))
	
	# Log test metrics to MLflow in OOF_Summary run
	with mlflow.start_run(run_name="Test_Summary"):
		mlflow.log_metric("test_auc_pr", test_auc_pr)
		mlflow.log_metric("test_auc_roc", test_auc_roc)
		mlflow.log_metric("test_mcc", test_mcc)
		mlflow.log_metric("test_accuracy", test_acc)
		mlflow.log_metric("test_f1_score", test_f1)
		mlflow.log_param("threshold_used", avg_threshold)
	
	# Save test metrics to file
	test_metrics_path = Path("src/models/trained_model") / model_name / "test_metrics.txt"
	test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
	
	with open(test_metrics_path, 'w') as f:
		f.write(f"Test Dataset Metrics for Model: {model_name}\n")
		f.write(f"Number of Test Samples: {len(test_dataset)}\n")
		f.write(f"Number of Folds: {config.num_folds}\n")
		f.write(f"Threshold Used (Average from CV): {avg_threshold:.4f}\n")
		f.write("="*80 + "\n\n")
		f.write("Test Metrics:\n")
		f.write("-"*80 + "\n")
		f.write(f"  AUC_PR: {test_auc_pr:.4f}\n")
		f.write(f"  AUC_ROC: {test_auc_roc:.4f}\n")
		f.write(f"  Accuracy: {test_acc:.4f}\n")
		f.write(f"  F1_score: {test_f1:.4f}\n")
		f.write(f"  MCC: {test_mcc:.4f}\n")
	
	print(f"\nTest Results:")
	print(f"  AUC_PR: {test_auc_pr:.4f}")
	print(f"  AUC_ROC: {test_auc_roc:.4f}")
	print(f"  Accuracy: {test_acc:.4f}")
	print(f"  F1_score: {test_f1:.4f}")
	print(f"  MCC: {test_mcc:.4f}")
	print(f"\nTest metrics saved to: {test_metrics_path}")
	print("="*80 + "\n")
	
	return all_fold_metrics



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