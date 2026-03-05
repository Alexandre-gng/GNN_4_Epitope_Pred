import os
import sys

import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models", "GAT"))

from GAT_func import TrainingConfig
from GAT_CV import cross_validate_gat


print(f"Project root directory: {PROJECT_ROOT}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_view_split(view_name: str, split_name: str):
    """Load one split for a given graph view from data/epitope3d/graph_list."""
    split_to_suffix = {
        "TRAIN": "TRAIN",
        "TEST": "TEST",
        "VAL": "VAL",
    }

    supported_views = {"spatial_1", "spatial_2", "sequential_1", "sequential_2"}
    if view_name not in supported_views:
        raise ValueError(f"Unknown view '{view_name}'. Supported: {sorted(supported_views)}")
    if split_name not in split_to_suffix:
        raise ValueError(f"Unknown split '{split_name}'. Supported: {list(split_to_suffix.keys())}")

    file_name = f"epitope3d_{split_to_suffix[split_name]}.pt"
    file_path = os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", view_name, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Graph split not found: {file_path}")
    dataset = torch.load(file_path)
    return normalize_dataset_for_gat(dataset)


def _recover_node_attrs_tensor(data):
    """Recover node features from legacy spatial2 format where node_attrs may be wrapped/callable."""
    if not hasattr(data, "keys") or "node_attrs" not in data:
        return None

    node_attrs = data["node_attrs"]
    if isinstance(node_attrs, torch.Tensor):
        return node_attrs

    if callable(node_attrs):
        owner = getattr(node_attrs, "__self__", None)
        if owner is not None and hasattr(owner, "keys") and "node_attrs" in owner:
            owner_node_attrs = owner["node_attrs"]
            if isinstance(owner_node_attrs, torch.Tensor):
                if hasattr(data, "train_mask") and data.train_mask is None and hasattr(owner, "train_mask"):
                    data.train_mask = owner.train_mask
                elif not hasattr(data, "train_mask") and hasattr(owner, "train_mask"):
                    data.train_mask = owner.train_mask
                return owner_node_attrs
    return None


def _recover_attr_tensor(data, keys):
    for key in keys:
        value = getattr(data, key, None)
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(data, "keys") and key in data:
            value = data[key]
            if isinstance(value, torch.Tensor):
                return value
    return None


def normalize_dataset_for_gat(dataset):
    """Normalize graph objects so GAT always receives x, y and optional edge_attr tensors."""
    for graph in dataset:
        x = getattr(graph, "x", None)
        if not isinstance(x, torch.Tensor):
            recovered_x = _recover_node_attrs_tensor(graph)
            if isinstance(recovered_x, torch.Tensor):
                graph.x = recovered_x

        if not isinstance(getattr(graph, "x", None), torch.Tensor):
            raise ValueError(
                f"Could not recover node features for graph '{getattr(graph, 'name', 'unknown')}'. "
                f"Available keys: {list(graph.keys()) if hasattr(graph, 'keys') else []}"
            )

        if hasattr(graph, "keys") and "node_attrs" in graph:
            del graph["node_attrs"]

        y = _recover_attr_tensor(graph, ["y", "label", "labels", "target"])
        if isinstance(y, torch.Tensor):
            graph.y = y
        if not isinstance(getattr(graph, "y", None), torch.Tensor):
            raise ValueError(
                f"Could not recover labels for graph '{getattr(graph, 'name', 'unknown')}'. "
                f"Available keys: {list(graph.keys()) if hasattr(graph, 'keys') else []}"
            )

        edge_attr = _recover_attr_tensor(graph, ["edge_attr", "edge_weight", "edge_weights"])
        if isinstance(edge_attr, torch.Tensor):
            graph.edge_attr = edge_attr

    return dataset


def get_node_features(data):
    """Return node feature tensor from x or node_attrs (supports legacy callable wrappers)."""
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

    return None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
roh1_train = load_view_split("spatial_1", "TRAIN")
roh1_val = load_view_split("spatial_1", "TEST")
roh1_blind_test = load_view_split("spatial_1", "VAL")
print(f"Loaded spatial_1  train: {len(roh1_train)}, val: {len(roh1_val)}, test: {len(roh1_blind_test)}")

roh2_train = load_view_split("spatial_2", "TRAIN")
roh2_val = load_view_split("spatial_2", "TEST")
roh2_blind_test = load_view_split("spatial_2", "VAL")
print(f"Loaded spatial_2  train: {len(roh2_train)}, val: {len(roh2_val)}, test: {len(roh2_blind_test)}")


# ====== Fixed hyperparameters ======
config = TrainingConfig(
    epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    num_folds=10,
    hidden_dim=128,
    heads=4,
    num_layers=3,
    dropout=0.1,
    patience=10,
    best_threshold=None,
)


# Name - Train dataset - Test dataset - use_edge_attr - Focal Loss - embedding_type
model_configs = [
    # spatial 1
    ("GAT_spatial1_bce_edge_w", roh1_train, roh1_val, roh1_blind_test, True, False, "ESM2+ESMIF1"),
    ("GAT_spatial1_bce", roh1_train, roh1_val, roh1_blind_test, False, False, "ESM2+ESMIF1"),
    # spatial 2
    ("GAT_spatial2_bce_edge_w", roh2_train, roh2_val, roh2_blind_test, True, False, "ESM2+ESMIF1"),
    ("GAT_spatial2_bce", roh2_train, roh2_val, roh2_blind_test, False, False, "ESM2+ESMIF1")
]

# Infer feature dimension robustly
sample_features = None
for graph in roh1_train:
    sample_features = get_node_features(graph)
    if isinstance(sample_features, torch.Tensor):
        break

if not isinstance(sample_features, torch.Tensor):
    raise ValueError(
        "Unable to infer input feature dimension from spatial_1 TRAIN split. "
        "No tensor found in 'x' or 'node_attrs'."
    )

input_dim = int(sample_features.shape[-1])
print("Feature dim:", input_dim)

for name, train_dataset, val_dataset, test_dataset, use_edge_attr, use_focal_loss, embedding_type in model_configs:
    print(f"\n{'='*80}")
    print(f"Cross-validating Model: {name}")
    print(f"Embedding: {embedding_type} | use_edge_attr: {use_edge_attr} | use_focal_loss: {use_focal_loss}")
    print(f"{'='*80}")

    oof = cross_validate_gat(
        train_data=train_dataset,
        val_data=val_dataset,
        blind_test_data=test_dataset,
        model_name=name,
        config=config,
        device=device,
        project_root=PROJECT_ROOT,
        use_focal_loss=use_focal_loss,
        use_edge_attr=use_edge_attr,
        in_channels=input_dim,
    )
    print(f"OOF results for {name}: {oof['oof_metrics']}")