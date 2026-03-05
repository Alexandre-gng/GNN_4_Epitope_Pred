"""
Cross-Validation for GAT model.

This module provides a function to perform k-fold cross-validation on the GAT model,
saving each fold's model state and computing Out-Of-Fold (OOF) metrics.
"""

import os
import json
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	f1_score,
	matthews_corrcoef,
	roc_auc_score,
)
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from GAT import GATv2Net
from GAT_func import (
	TrainingConfig,
	compute_predictions,
	evaluate,
	evaluate_w_threshold,
	find_best_threshold,
	infer_edge_dim,
	train_n_epochs,
)
from FOCAL_LOSS import FocalLoss


def _safe_metric(metric_fn, y_true, y_pred, default: float = 0.0) -> float:
	try:
		return float(metric_fn(y_true, y_pred))
	except Exception:
		return float(default)


def cross_validate_gat(
	train_data: List,
	val_data: List,
	blind_test_data: List,
	model_name: str,
	config: TrainingConfig,
	device: torch.device,
	project_root: str,
	use_focal_loss: bool = False,
	use_edge_attr: bool = False,
	in_channels: int | None = None,
) -> Dict:
	"""
	Perform k-fold cross-validation on GAT model and save OOF predictions and metrics.

	Data split strategy:
		- CV is performed on combined train + val datasets.
		- Each fold splits these samples into train/val subsets.
		- Blind test set (separate antigens) is used for final evaluation of each fold.
		- OOF predictions are computed on the CV validation folds only.
	"""
	def _extract_tensor(g, keys: List[str]) -> torch.Tensor | None:
		for key in keys:
			value = getattr(g, key, None)
			if torch.is_tensor(value):
				return value

			if hasattr(g, "keys") and key in g:
				value = g[key]
				if torch.is_tensor(value):
					return value
				if callable(value):
					owner = getattr(value, "__self__", None)
					if owner is not None and hasattr(owner, "keys") and key in owner:
						nested = owner[key]
						if torch.is_tensor(nested):
							return nested
		return None

	def _sanitize_graphs(graphs: List, split_name: str) -> List:
		cleaned = []
		dropped = []
		for idx, g in enumerate(graphs):
			x = _extract_tensor(g, ["x", "node_attrs"])
			y = _extract_tensor(g, ["y", "label", "labels", "target"])
			edge_index = _extract_tensor(g, ["edge_index"])
			edge_attr = _extract_tensor(g, ["edge_attr", "edge_weight", "edge_weights"])

			if not (torch.is_tensor(x) and torch.is_tensor(y) and torch.is_tensor(edge_index)):
				dropped.append(
					(
						idx,
						f"x={type(x)}, y={type(y)}, edge_index={type(edge_index)}",
					),
				)
				continue

			g.x = x
			g.y = y
			if torch.is_tensor(edge_attr):
				g.edge_attr = edge_attr
			if hasattr(g, "keys") and "node_attrs" in g:
				del g["node_attrs"]

			cleaned.append(g)

		if dropped:
			print(f"[WARN] Dropped {len(dropped)} malformed graphs in {split_name}; first 3: {dropped[:3]}")
		return cleaned

	train_data = _sanitize_graphs(train_data, "train_data")
	val_data = _sanitize_graphs(val_data, "val_data")
	blind_test_data = _sanitize_graphs(blind_test_data, "blind_test_data")
	if len(train_data) == 0:
		raise ValueError("All training graphs are malformed after sanitization.")

	combined_data = train_data + val_data
	if len(combined_data) == 0:
		raise ValueError("Combined train+val dataset is empty.")

	if in_channels is None:
		sample = combined_data[0]
		if not hasattr(sample, "x") or not torch.is_tensor(sample.x):
			raise ValueError("Unable to infer in_channels: sample has no 'x' tensor.")
		in_channels = int(sample.x.shape[-1])

	print(f"\n{'='*80}")
	print(f"CROSS-VALIDATION SETUP — {model_name}")
	print(f"{'='*80}")
	print(f"Total samples for CV (train + val): {len(combined_data)}")
	print(f"Number of folds: {config.num_folds}")
	print(f"Samples per fold (approx): {len(combined_data) // config.num_folds}")
	print(f"Blind test set: {len(blind_test_data)} separate antigens")
	print(f"Input feature dim: {in_channels}")
	print(f"{'='*80}\n")

	kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=42)
	all_oof_preds: List[np.ndarray] = []
	all_oof_labels: List[np.ndarray] = []
	fold_results: List[Dict] = []

	model_dir = os.path.join(project_root, "src", "models", "trained_model", model_name)
	os.makedirs(model_dir, exist_ok=True)

	for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(len(combined_data))), start=1):
		print(f"\n{'='*80}")
		print(f"Fold {fold_idx}/{config.num_folds}")
		print(f"{'='*80}")
		print(f"  Training samples:          {len(train_indices)}")
		print(f"  Validation samples (OOF):  {len(val_indices)}")
		print(f"  Blind test samples:        {len(blind_test_data)}")

		train_fold = [combined_data[i] for i in train_indices]
		val_fold = [combined_data[i] for i in val_indices]

		train_loader = DataLoader(train_fold, batch_size=config.batch_size, shuffle=True)
		val_loader = DataLoader(val_fold, batch_size=config.batch_size, shuffle=False)
		blind_test_loader = DataLoader(blind_test_data, batch_size=config.batch_size, shuffle=False)

		edge_dim = infer_edge_dim(train_fold) if use_edge_attr else None

		model = GATv2Net(
			input_dim=in_channels,
			hidden_dim=config.hidden_dim,
			output_dim=1,
			num_layers=config.num_layers,
			heads=config.heads,
			concat=True,
			residual=True,
			dropout=config.dropout,
			edge_dim=edge_dim,
		).to(device)

		optimizer = torch.optim.AdamW(
			model.parameters(),
			lr=config.learning_rate,
			weight_decay=config.weight_decay,
		)

		if use_focal_loss:
			criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
		else:
			criterion = nn.BCEWithLogitsLoss()

		best_model_state = train_n_epochs(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			optimizer=optimizer,
			criterion=criterion,
			name_model=f"{model_name}_fold_{fold_idx}",
			device=device,
			use_edge_attr=use_edge_attr,
			n_epochs=config.epochs,
			patience=config.patience,
			use_mlflow=False,
		)

		model.load_state_dict(best_model_state)

		print(f"\n>>> Verifying loaded best model for fold {fold_idx}...")
		verify_val_loss, verify_val_ap, verify_val_roc = evaluate(
			model, val_loader, criterion, device, use_edge_attr=use_edge_attr
		)
		verify_test_loss, verify_test_ap, verify_test_roc = evaluate(
			model, blind_test_loader, criterion, device, use_edge_attr=use_edge_attr
		)
		print(f">>> Verification — Val AP: {verify_val_ap:.4f}, Val ROC: {verify_val_roc:.4f}")
		print(f">>> Verification — Blind Test AP: {verify_test_ap:.4f}, Blind Test ROC: {verify_test_roc:.4f}")

		best_threshold = find_best_threshold(
			model, val_loader=val_loader, device=device, use_edge_attr=use_edge_attr
		)
		print(f"Best threshold for fold {fold_idx}: {best_threshold:.4f}")

		oof_preds, oof_labels, _ = compute_predictions(
			model, val_loader, criterion, device, use_edge_attr=use_edge_attr
		)
		if oof_preds.size > 0 and oof_labels.size > 0:
			all_oof_preds.append(oof_preds)
			all_oof_labels.append(oof_labels)

		val_loss, val_auc_pr, val_auc_roc, val_mcc, val_acc, val_f1 = evaluate_w_threshold(
			model, val_loader, best_threshold, criterion, device, use_edge_attr
		)
		test_loss, test_auc_pr, test_auc_roc, test_mcc, test_acc, test_f1 = evaluate_w_threshold(
			model, blind_test_loader, best_threshold, criterion, device, use_edge_attr
		)

		val_ap_diff = abs(val_auc_pr - verify_val_ap)
		test_ap_diff = abs(test_auc_pr - verify_test_ap)
		if val_ap_diff > 0.001 or test_ap_diff > 0.001:
			print("  WARNING: Metrics mismatch detected!")
			print(f"   Val AP diff: {val_ap_diff:.6f}  |  Test AP diff: {test_ap_diff:.6f}")
		else:
			print("  Metrics verification passed (differences < 0.001)")

		fold_result = {
			"fold": fold_idx,
			"edge_dim": edge_dim,
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

		checkpoint = {
			"fold": fold_idx,
			"model_state_dict": best_model_state,
			"config": asdict(config),
			"threshold": float(best_threshold),
			"metrics": fold_result,
		}
		fold_path = os.path.join(model_dir, f"fold_{fold_idx}.pt")
		torch.save(checkpoint, fold_path)
		print(f"Saved fold {fold_idx} checkpoint to {fold_path}")

	print(f"\n{'='*80}")
	print("STEP 4: Computing Out-Of-Fold (OOF) Metrics")
	print(f"{'='*80}")

	if len(all_oof_preds) == 0 or len(all_oof_labels) == 0:
		raise ValueError("No OOF predictions collected.")

	all_oof_preds_arr = np.concatenate(all_oof_preds)
	all_oof_labels_arr = np.concatenate(all_oof_labels).astype(int)

	if len(np.unique(all_oof_labels_arr)) < 2:
		raise ValueError(f"OOF labels contain only one class: {np.unique(all_oof_labels_arr)}")

	oof_auc_pr = _safe_metric(average_precision_score, all_oof_labels_arr, all_oof_preds_arr)
	oof_auc_roc = _safe_metric(roc_auc_score, all_oof_labels_arr, all_oof_preds_arr)

	best_oof_threshold = 0.5
	best_oof_mcc = -1.0
	for threshold in np.arange(0.0, 1.01, 0.01):
		binary_preds = (all_oof_preds_arr >= threshold).astype(int)
		mcc_score = _safe_metric(matthews_corrcoef, all_oof_labels_arr, binary_preds, default=-1.0)
		if mcc_score > best_oof_mcc:
			best_oof_mcc = mcc_score
			best_oof_threshold = float(threshold)

	oof_binary_preds = (all_oof_preds_arr >= best_oof_threshold).astype(int)
	oof_mcc = _safe_metric(matthews_corrcoef, all_oof_labels_arr, oof_binary_preds)
	oof_f1 = _safe_metric(f1_score, all_oof_labels_arr, oof_binary_preds)
	oof_acc = _safe_metric(accuracy_score, all_oof_labels_arr, oof_binary_preds)

	def _mean(scope: str, key: str) -> float:
		return float(np.mean([f[scope][key] for f in fold_results]))

	def _sem(scope: str, key: str) -> float:
		vals = [f[scope][key] for f in fold_results]
		if len(vals) < 2:
			return 0.0
		return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

	avg_val_metrics = {k: _mean("validation", k) for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")}
	avg_test_metrics = {k: _mean("blind_test", k) for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")}
	sem_val_metrics = {k: _sem("validation", k) for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")}
	sem_test_metrics = {k: _sem("blind_test", k) for k in ("loss", "auc_pr", "auc_roc", "mcc", "f1", "accuracy")}

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
	with open(oof_metrics_path, "w", encoding="utf-8") as f:
		json.dump(oof_metrics, f, indent=4)

	print(f"Saved OOF metrics to {oof_metrics_path}")
	return oof_metrics
