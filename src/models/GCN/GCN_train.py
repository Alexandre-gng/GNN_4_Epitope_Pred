"""
GCN_train.py — Train a single GCN model on one graph view of the Epitope3D dataset.

Usage:
    cd <PROJECT_ROOT>
    python src/models/GCN/GCN_train.py
"""

import os
import sys
from dataclasses import asdict

import torch
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models", "GCN"))

from GCN import PyGGCNModel
from GCN_func import TrainingConfig
from GCN_CV import cross_validate_gcn

# Also make FocalLoss available (shared across models)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models", "MGAT"))
from FOCAL_LOSS import FocalLoss

print(f"Project root directory: {PROJECT_ROOT}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_view_split(view_name: str, split_name: str):
    """Load one split for a given graph view from data/epitope3d/graph_list (same source as EGNN)."""
    split_to_suffix = {
        "TRAIN": "TRAIN",
        "TEST": "TEST",
        "VAL": "VAL",
    }

    supported_views = {"spatial_1", "spatial_2"}
    if view_name not in supported_views:
        raise ValueError(f"Unknown view '{view_name}'. Supported: {sorted(supported_views)}")
    if split_name not in split_to_suffix:
        raise ValueError(f"Unknown split '{split_name}'. Supported: {list(split_to_suffix.keys())}")

    file_name = f"epitope3d_{split_to_suffix[split_name]}.pt"
    file_path = os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", view_name, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Graph split not found: {file_path}")
    dataset = torch.load(file_path)
    return normalize_dataset_features(dataset)


def _recover_node_attrs_tensor(data):
    """Recover node feature tensor even when node_attrs is a bound method (spatial_2 legacy files)."""
    if "node_attrs" not in data:
        return None

    node_attrs = data["node_attrs"]
    if isinstance(node_attrs, torch.Tensor):
        return node_attrs

    if callable(node_attrs):
        owner = getattr(node_attrs, "__self__", None)
        if owner is not None and "node_attrs" in owner:
            owner_node_attrs = owner["node_attrs"]
            if isinstance(owner_node_attrs, torch.Tensor):
                if hasattr(data, "train_mask") and data.train_mask is None and hasattr(owner, "train_mask"):
                    data.train_mask = owner.train_mask
                elif not hasattr(data, "train_mask") and hasattr(owner, "train_mask"):
                    data.train_mask = owner.train_mask
                return owner_node_attrs

    return None


def normalize_dataset_features(dataset):
    """Ensure each graph has tensor node features in x for stable batching/model forward."""
    for graph in dataset:
        x = getattr(graph, "x", None)
        if isinstance(x, torch.Tensor):
            if "node_attrs" in graph:
                del graph["node_attrs"]
            continue

        recovered = _recover_node_attrs_tensor(graph)
        if isinstance(recovered, torch.Tensor):
            graph.x = recovered
            if "node_attrs" in graph:
                del graph["node_attrs"]
            continue

        raise ValueError(
            f"Could not recover node features for graph '{getattr(graph, 'name', 'unknown')}'. "
            f"Available keys: {list(graph.keys())}"
        )

    return dataset


def get_node_features(data):
    """Return node feature tensor, prioritizing x and safely handling node_attrs collisions."""
    x = getattr(data, "x", None)
    if isinstance(x, torch.Tensor):
        return x

    if "node_attrs" in data:
        node_attrs = data["node_attrs"]
        if isinstance(node_attrs, torch.Tensor):
            return node_attrs

    raise ValueError(
        f"No tensor node features found. Available keys: {list(data.keys())}. "
        "Expected tensor in 'x' or 'node_attrs'."
    )


# ---------------------------------------------------------------------------
# Load data  — pick ONE view (change paths to test another view)
# ---------------------------------------------------------------------------
# Spatial 1
roh1_train = load_view_split("spatial_1", "TRAIN")
roh1_val   = load_view_split("spatial_1", "TEST")
roh1_test  = load_view_split("spatial_1", "VAL")
print(f"Loaded  train: {len(roh1_train)},  val: {len(roh1_val)},  test: {len(roh1_test)}  graphs.")
# Spatial 
roh2_train = load_view_split("spatial_2", "TRAIN")
roh2_val   = load_view_split("spatial_2", "TEST")
roh2_test  = load_view_split("spatial_2", "VAL")
print(f"Loaded  train: {len(roh2_train)},  val: {len(roh2_val)},  test: {len(roh2_test)}  graphs.")


# Infer feature dimensions from the first available sample with valid features
sample = next((g for g in roh1_train if isinstance(getattr(g, "x", None), torch.Tensor)), None)
if sample is None:
    sample = next((g for g in roh1_train if "node_attrs" in g and isinstance(g["node_attrs"], torch.Tensor)), None)
if sample is None:
    raise ValueError("Unable to infer input_dim from spatial_1 TRAIN split: no tensor features found in 'x' or 'node_attrs'.")

input_dim = get_node_features(sample).shape[-1]
print(f"Feature dim: {input_dim}")


def infer_edge_dim(dataset) -> int:
    """Return 1 if edge_attr exists and is scalar/1-d, else 0."""
    for data in dataset:
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            if data.edge_attr.dim() == 1:
                return 1
            return data.edge_attr.size(-1)
    return 0

print(f"Edge attribute dimension: {infer_edge_dim(roh1_train)}")

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
config = TrainingConfig(
    epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    n_hidden=128,
    n_layers=3,
    n_edge_attr=infer_edge_dim(roh1_train),
    dropout=0.3,
    patience=10,
    alpha=0.25,
    gamma=2.0,
    best_threshold=None,
)

print(f"Config: {asdict(config)}")

# ---------------------------------------------------------------------------
# Model configurations to train
# ---------------------------------------------------------------------------
# (name, train_data, val_data, test_data, use_focal_loss, use_edge_weights, embedding_type)
model_configs = [
    ("GCN_spatial1_bce_edge_weights", roh1_train, roh1_val, roh1_test, False,  True, "ESM2+ESMIF1"),
    ("GCN_spatial1_bce",   roh1_train, roh1_val, roh1_test, False, False, "ESM2+ESMIF1"),
    ("GCN_spatial2_bce_edge_weights", roh2_train, roh2_val, roh2_test, False,  True, "ESM2+ESMIF1"),
    ("GCN_spatial2_bce",   roh2_train, roh2_val, roh2_test, False, False, "ESM2+ESMIF1"),
]



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for name, train_data, val_data, test_data, use_focal_loss, use_edge_weights, embedding_type in model_configs:
    print(f"\n{'=' * 80}")
    print(f"Training Model: {name}  |  Focal Loss: {use_focal_loss}  |  Edge Weights: {use_edge_weights}  |  Embedding: {embedding_type}")
    print(f"{'=' * 80}")

    config.n_edge_attr = use_edge_weights
    oof = cross_validate_gcn(
        train_data=train_data,
        val_data=val_data,
        blind_test_data=test_data,
        model_name=name,
        config=config,
        device=device,
        project_root=PROJECT_ROOT,
        use_focal_loss=use_focal_loss,
        use_edge_weights=use_edge_weights,
        use_mask=True,
        in_channels=input_dim,
    )
    print(f"OOF results for {name}: {oof}")
